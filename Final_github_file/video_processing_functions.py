import os, re
import numpy as np
import pandas as pd
from math import sqrt
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pylab

def threshold_image(img,threshold_val=50):
    """
    only keeps pixel values over "threshold_val". This helps tremendously in finding the wrinkles, and allows us to choose smaller 
    n_cluster
    """
#     timg = np.asarray(img)
    thresh = img > threshold_val
    out = thresh * img
#     img = Image.fromarray(out)
    return out


def bf_size(image, threshold_val=30):
    a = image.max(axis=2)
    tmp = a > threshold_val
    tmp_im = a * tmp
    return int(np.count_nonzero(tmp_im))