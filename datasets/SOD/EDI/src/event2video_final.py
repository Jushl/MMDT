from .fibosearch import fibosearch
from .TVnorm import TVnorm
import numpy as np


def warpingBlur2If(delta, blur, x_o, y_o, pol_o, t_o, expstart, expend, Estart):
    t1 = expstart
    t3 = Estart
    t2 = expend
    x, y, pol, t = x_o, y_o, pol_o, t_o
    idx = (t >= t3) & (t <= t2)
    idx = idx.astype(np.int)

    x = x[idx == 1]
    y = y[idx == 1]
    pol = pol[idx == 1]
    t = t[idx == 1]

    h, w = blur.shape
    dim = 1
    blur = blur.reshape(h, w, dim)

    et = np.zeros((h, w))
    Et = np.zeros((h, w))
    edgemap = np.zeros((h, w))
    thr = delta / 255
    intgltime = np.ones((h, w)) * t3

    for i in range(len(y)):
        u = int(y[i])
        v = int(x[i])
        pk = pol[i]

        Et[u, v] = Et[u, v] + np.exp(et[u, v]) * (t[i] - intgltime[u, v])
        et[u, v] = et[u, v] + pk * thr
        # edgemap[u, v] = edgemap[u, v] + pk * thr * np.exp(-(t[i] - intgltime[u, v]))
        intgltime[u, v] = t[i]

    Et = Et + np.exp(et) * (t2 - intgltime)
    newEt1 = Et.copy()

    x, y, pol, t = x_o, y_o, pol_o, t_o
    idx = (t >= t1) & (t < t3)
    idx = idx.astype(np.int)

    x = x[idx == 1]
    y = y[idx == 1]
    pol = pol[idx == 1]
    t = t[idx == 1]

    et = np.zeros((h, w))
    Et = np.zeros((h, w))
    thr = delta / 255
    intgltime = np.ones((h, w)) * t3

    for i in range(len(y) - 1, -1, -1):
        u = int(y[i])
        v = int(x[i])
        pk = -pol[i]

        Et[u, v] = Et[u, v] + np.exp(et[u, v]) * np.abs(t[i] - intgltime[u, v])
        et[u, v] = et[u, v] + pk * thr
        # edgemap[u, v] = edgemap[u, v] + pk * thr * np.exp((t[i] - intgltime[u, v]))
        intgltime[u, v] = t[i]

    Et = Et + np.exp(et) * np.abs(t1 - intgltime)
    newEt2 = Et.copy()
    newEt = (newEt1 + newEt2) / (t2 - t1)
    I = blur / np.tile(newEt.reshape(h, w, 1), (1, 1, dim))
    I[I > 1.1] = 1.1

    return I, edgemap


def estdelta(blur, x, y, pol, t, eventstart, eventend, Estart):
    imclean = lambda delta: warpingBlur(delta, blur, x, y, pol, t, eventstart, eventend, Estart)
    fun = lambda delta: imclean(delta)

    ndelta = fibosearch(fun, 1, 255, 30)
    ndelta = min(189.5, max(100, ndelta))
    return ndelta


def warpingBlur(delta, blur, x, y, pol, t, eventstart, eventend, Estart):
    I, edgemap = warpingBlur2If(delta, blur, x, y, pol, t, eventstart, eventend, Estart)
    L = I
    Ltv = TVnorm(L, edgemap)
    return Ltv


def event2video_final(blur, x, y, pol, t, eventstart, eventend, exptime, v_length, delta):
    I_video = [None] * v_length
    deltaT = [0] * v_length
    tsample = exptime / v_length

    for ts in range(v_length):
        ts += 1
        Estart = eventstart + tsample * (ts - 1)

        # delta = estdelta(blur, x, y, pol, t, eventstart, eventend, Estart)
        It, _ = warpingBlur2If(delta, blur, x, y, pol, t, eventstart, eventend, Estart)
        deltaT[ts - 1] = delta
        I_video[ts - 1] = It

    return I_video, deltaT