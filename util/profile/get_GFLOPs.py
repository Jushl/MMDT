import torch.nn as nn
from thop.profile import prRed
import torch
# from thop.profile import register_hooks
from util.profile.my_profile import register_hooks


def profile(model: nn.Module, input, custom_ops=None, verbose=True, ret_layer_info=False, report_missing=False, ):
    handler_collection = {}
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}
    if report_missing:
        verbose = True

    def calculate_parameters(param_list):
        total_params = 0
        for p in param_list:
            total_params += torch.DoubleTensor([p.nelement()])
        return total_params

    def count_parameters(m, x, y):
        total_params = 0
        for p in m.parameters():
            total_params += torch.DoubleTensor([p.numel()])
        m.total_params[0] = calculate_parameters(m.parameters())

    def add_hooks(m: nn.Module):
        m.register_buffer("total_ops", torch.zeros(1, dtype=torch.float64))   # register_buffer:为每个模块添加ops和params计数器,初始化为0。
        m.register_buffer("total_params", torch.zeros(1, dtype=torch.float64))

        m_type = type(m)
        fn = None
        if m_type in custom_ops:  # custom_ops:{}
            fn = custom_ops[m_type]
            if m_type not in types_collection and verbose:
                print("[INFO] Customize rule %s() %s." % (fn.__qualname__, m_type))
        elif m_type in register_hooks:  # 已经注册过的hooks例如conv2d，relu这种， 看哪一个没注册钩子给它注册了
            fn = register_hooks[m_type]
            if m_type not in types_collection and verbose:
                print("[INFO] Register %s() for %s." % (fn.__qualname__, m_type))
        else:
            if m_type not in types_collection and report_missing:
                prRed("[WARN] Cannot find rule for %s. Treat it as zero Macs and zero Params." % m_type)
        if fn is not None:
            handler_collection[m] = (m.register_forward_hook(fn), m.register_forward_hook(count_parameters),)
        types_collection.add(m_type)

    prev_training_status = model.training
    model.eval()
    model.apply(add_hooks)
    with torch.no_grad():
        model(*input)

    def dfs_count(module: nn.Module, prefix="\t") -> (int, int):
        global encoder_ops, decoder_ops, tgt_ops, encoder_energys, decoder_energys, tgt_energys
        total_ops, total_params = module.total_ops.item(), 0
        ret_dict = {}
        for n, m in module.named_children():
            next_dict = {}
            if m in handler_collection and not isinstance(m, (nn.Sequential, nn.ModuleList)):
                m_ops, m_params = m.total_ops.item(), m.total_params.item()
            else:
                m_ops, m_params, next_dict = dfs_count(m, prefix=prefix + "\t")
            ret_dict[n] = (m_ops, m_params, next_dict)
            total_ops += m_ops
            total_params += m_params

            # 计算enegy： FLOPs x  单位焦耳数       FLOPs：执行加、减、乘、除等基本算术运算所需的总操作次数
            if n == 'encoder':
                encoder_ops = total_ops
                print('encoder_GFLOPs:', total_ops / 1e9)
            if n == 'decoder':
                decoder_ops = total_ops - encoder_ops
                print('decoder_GFLOPs:', decoder_ops / 1e9)
            if n == 'input_proj':
                tgt_ops = total_ops - encoder_ops - decoder_ops
                print('tgt_GFLOPs:', tgt_ops / 1e9)
            if n == 'backbone':
                backbone_ops = total_ops - tgt_ops - encoder_ops - decoder_ops
                print('backbone_GFLOPs:', backbone_ops / 1e9)
            # print('module_name:', n, '______________Total_GFLOPs:', total_ops / 1e9, '______________Total_energy:', total_energy / 1e9)
        return total_ops, total_params, ret_dict

    total_ops, total_params, ret_dict = dfs_count(model)
    model.train(prev_training_status)
    for m, (op_handler, params_handler) in handler_collection.items():
        op_handler.remove()
        params_handler.remove()
        m._buffers.pop("total_ops")
        m._buffers.pop("total_params")
    if ret_layer_info:
        return total_ops, total_params, ret_dict
    print("Total Parameters：", total_params / 1e6)
    print("Total GFLOPS：", total_ops / 1e9)

