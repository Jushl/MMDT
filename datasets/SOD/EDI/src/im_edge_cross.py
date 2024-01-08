import numpy as np
import cv2


def edge_sobel(img):
    edges = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
    return edges


def im_edge_cross(Img, edgemap):
    I_edge = edge_sobel(Img)
    e_edge = edge_sobel(edgemap)
    p_cros = I_edge * e_edge
    return p_cros


def stretch(im):
    vmax = np.max(im)
    vmin = np.min(im)
    scale = 1.0 / (vmax - vmin)
    a = (im - vmin) * scale
    return a
