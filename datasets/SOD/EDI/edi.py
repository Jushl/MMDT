import os.path
import numpy as np
import cv2
from .src.event2video_final import event2video_final


def mat2gray(image):
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val != min_val:
        image = (image - min_val) / (max_val - min_val)
    return image


def blur_to_sharp(png_path, npy_path, v_length=100, delta=100):
    timescale = 1e6

    if isinstance(npy_path, str):
        data = np.load(npy_path)
    else:
        data = npy_path  # t,x,y,p

    x_o, y_o, pol_o, t_o = data[:, 0], data[:, 1], data[:, 3].copy(), data[:, 2]
    pol_o[pol_o == 0] = -1
    t_o = np.mod(t_o, 1e8)
    t_o /= timescale
    x, y, pol, t = x_o, y_o, pol_o, t_o

    img = cv2.imread(png_path, 0)

    blur = np.asarray(img)
    blur = mat2gray(blur)

    eventstart = t_o[0]
    eventend = t_o[-1]
    exptime = eventend - eventstart

    I_video, deltaT = event2video_final(blur, x, y, pol, t, eventstart, eventend, exptime, v_length, delta)

    for i in range(len(I_video)):
        I_video[i] = mat2gray(I_video[i]) * 255

    return I_video, deltaT, delta




