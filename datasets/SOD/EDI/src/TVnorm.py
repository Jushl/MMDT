import numpy as np
from .im_edge_cross import im_edge_cross


def rgb2gray(rgb):
    gray = np.dot(rgb, [0.2989, 0.5870, 0.1140])
    return gray


def dxp(u):
    dx = np.hstack((u[:, 1:], u[:, -1:])) - u
    return dx


def dyp(u):
    dy = np.vstack((u[1:, :], u[-1:, :])) - u
    return dy


def TVnorm(L, edgemp):
    lambda_ = 0.2
    h, w, dim = L.shape
    L = L.reshape(h, w, dim)
    if dim == 3:
        L = rgb2gray(L)
    L_x = dxp(L)
    L_y = dyp(L)
    Ltv = np.sum(np.sqrt(L_x ** 2 + L_y ** 2))
    p_cross = im_edge_cross(L, edgemp)
    Ltv = lambda_ * Ltv - np.sum(p_cross)
    return Ltv

