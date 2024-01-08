import torch
from torch import Tensor
import numpy as np


def time_entropy_encoder(time_data):
    diff = np.diff(time_data)
    diff = np.insert(diff, 0, 0)
    diff_mean = np.mean(diff)
    diff_std = np.std(diff)
    diff = (diff - diff_mean) ** 2 / diff_std
    time_entropy = np.exp(-diff)
    return time_entropy


def time_weight_encoder(time_data):
    unique, counts = np.unique(time_data, return_counts=True)
    result = np.zeros_like(time_data)
    for u, c in zip(unique, counts):
        result[time_data == u] = c
    max_count = np.max(result)
    time_weight = result / max_count
    return time_weight


def time_sequence_encoder(time_data, Kr=1.05):
    time_data[time_data == 0] = 1
    time_sequence = np.power(Kr, -np.log(np.abs(time_data)))
    time_sequence = time_sequence[np.argsort(time_sequence)]
    return time_sequence


def process_data(data, n_segments):
    per_num_seg = len(data) // n_segments
    outputs = torch.zeros(n_segments, 260, 346)
    max_limit = (data[:, 0].max(), data[:, 1].max())
    assert max_limit <= (260, 346), 'Exceeds the limit!!!'
    for i in range(n_segments):
        start = i * per_num_seg
        end = (i + 1) * per_num_seg if i < n_segments - 1 else len(data)
        seg_data = data[start:end]
        for j, row in enumerate(seg_data):
            coords = [int(row[0]), int(row[1])]
            if coords != [0, 0]:
                outputs[i, coords[0], coords[1]] += row[3].astype(np.float32)
    return outputs


def stream_to_tensor(data=None, num_channel=None):
    if isinstance(data, str): data = np.load(data)
    data[:, 2] -= data[0, 2]
    data[:, 3] -= 0.5
    time_data, polarity = data[:, 2], data[:, 3]
    time_entropy = time_entropy_encoder(time_data)
    time_sequence = time_sequence_encoder(time_data, Kr=1.05)
    # time_weight = time_weight_encoder(time_data)
    time_data = time_entropy + time_sequence  # + time_weight
    polarity *= time_data
    data[:, 3] = polarity
    data[:, [1, 0]] = data[:, [0, 1]]
    event_stream_tensor = process_data(data, num_channel)
    return event_stream_tensor


def to_tensor(pic, num_channel=1, mapping=True) -> Tensor:
    default_float_dtype = torch.get_default_dtype()
    event = stream_to_tensor(pic, num_channel).contiguous()
    event = event.view(1, -1)
    pos_norm_event = event[event > 0]
    pos_mean = pos_norm_event.mean()
    neg_norm_event = event[event < 0]
    neg_mean = neg_norm_event.mean()

    event = torch.clamp(event, min=neg_mean * 5, max=pos_mean * 5)

    pos_norm_event = event[event > 0]
    pos_mean = pos_norm_event.mean()
    pos_var = pos_norm_event.var()
    neg_norm_event = event[event < 0]
    neg_mean = neg_norm_event.mean()
    neg_var = neg_norm_event.var()

    event = torch.clamp(event, min=neg_mean - 3 * neg_var, max=pos_mean + 3 * pos_var)
    max, min = torch.max(event, dim=1)[0], torch.min(event, dim=1)[0]
    event[event > 0] /= max
    event[event < 0] /= abs(min)
    normalized_event = event.view(num_channel, 260, 346)

    # if mapping:
    #     mapped_event = torch.zeros_like(normalized_event)
    #     mapped_event[normalized_event < 0] = normalized_event[normalized_event < 0] * 128 + 128
    #     mapped_event[normalized_event >= 0] = normalized_event[normalized_event >= 0] * 127 + 128
    #     normalized_event = mapped_event
    #
    # if isinstance(normalized_event, torch.ByteTensor):
    #     return normalized_event.to(dtype=default_float_dtype)
    # else:
    #     return normalized_event

    if mapping:  # pos and neg
        pos_mapped_event = torch.zeros_like(normalized_event)
        pos_mapped_event[normalized_event >= 0] = normalized_event[normalized_event >= 0] * 255

        neg_mapped_event = torch.zeros_like(normalized_event)
        neg_mapped_event[normalized_event < 0] = abs(normalized_event[normalized_event < 0] * 255)

        mapped_event = torch.stack([pos_mapped_event, neg_mapped_event], dim=0)
    else:
        mapped_event = normalized_event

    if isinstance(normalized_event, torch.ByteTensor):
        return mapped_event.to(dtype=default_float_dtype)
    else:
        return mapped_event


class ToTensor(object):
    def __init__(self, time_steps):
        self.time_steps = time_steps

    def __call__(self, img, target):
        return to_tensor(img, self.time_steps), target
