import os, re
import numpy as np
import pandas as pd
from math import sqrt
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pylab


def polar_coord(X):
    """
    X: image (array format) having cartesian coordinates

    returns: the initial image transformed into polar coordinates 
    """
    r = np.sqrt(((X.shape[0]/2.0)**2.0)+((X.shape[1]/2.0)**2.0)) #get radius
    X_polar = cv2.linearPolar(X,(X.shape[0]/2, X.shape[1]/2),r, cv2.WARP_FILL_OUTLIERS)
    X_polar = X_polar.astype(np.uint8)
    return(X_polar)


def cartesian_coord(X):
    """
    X: image (array format) 

    returns: the initial image transformed into cartesian coordinates 
    """
    r = np.sqrt(((X.shape[0]/2.0)**2.0)+((X.shape[1]/2.0)**2.0)) #get radius
    X_cartesian=cv2.linearPolar(X, (X.shape[0]/2, X.shape[1]/2),r, cv2.WARP_INVERSE_MAP)
    X_cartesian = X_cartesian.astype(np.uint8)
    return(X_cartesian)


def get_wrinkles(img, resize = 500, is_movie = False, is_red = True, background_is_black = True, alpha=80, 
                 beta=100, mask=40):
    
    """
    img: initial image

    returns: the labels of the pixels and the image colored by the classes
    """
    # convert into image if is array
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        pass
    
    # Resize

    size = (resize, resize)
    img_resize       = img.resize(size, Image.ANTIALIAS)#resize
    img_array_resize = np.asarray(img_resize)
    
    
    
    if is_movie or not is_red :
        
        a = img_array_resize.mean(axis=2) < alpha
        b = img_array_resize.max(axis=2) > beta
        fil = a*b
        classes = fil * img_array_resize.max(axis=2)
        classes[classes > 0] = 1
        labels = np.reshape(classes, resize * resize)
        
    else:

        # Reshape
        w,h,d = tuple(img_array_resize.shape)
        X     = np.reshape(img_array_resize, (w*h,d))

        # Add radial distance as feature
        y_axis = np.array([])
        for i in np.arange(resize):
            y_axis = np.append(y_axis, np.arange(resize))

        y_axis          = np.reshape(y_axis, (resize**2, 1))
        x_axis          = np.reshape(np.repeat(np.arange(resize), resize), (resize**2, 1))
        radial_distance = np.sqrt(np.abs(x_axis - (resize / 2))**2 + np.abs(y_axis - (resize / 2))**2)
        X               = np.append(X, radial_distance, axis=1)

        # Determine labels and classes
        img_array = np.asarray(img)
        w,h,d     = tuple(img_array.shape)
        labels = ((X[:,0] > 100) * 1) & np.abs((1 - (X[:,1] / X[:,2])) < 0.75) & (X[:,1] < 80) & (X[:,2] < 80) & ((X[:,0]/ (X[:,1] +  X[:,2])) > 1.0) 
        classes   = np.reshape(labels, size)
        
    # apply black mask to background    
    lower_black = np.array([0,0,0], dtype = "uint16")
    upper_black = np.array([mask,mask,mask], dtype = "uint16")
    black_mask = cv2.inRange(img_array_resize, lower_black, upper_black)

    
    if background_is_black :
        classes[black_mask > 0] = 3
        
    return(labels, classes)


def perc_wrinkled(wrinkle_labels):
    
    """
    input : wrinkle_labels from the get_wrinkles method
    
    returns: the percent of the surface of the biofilm covered by wrinkles
    """
    return round(100*sum(np.array(wrinkle_labels) == 1) / (sum(np.array(wrinkle_labels) == 0) + sum(np.array(wrinkle_labels) == 1)), 2)



def detect_spokes(img, img_wrinkle = None, resize = 500, is_movie = False, is_red = True, rho = 1, theta = np.pi/180, minLineLength = 10, maxLineGap = 2 , threshold = 10):
    """
    input img
    
    returns: the initial images with the lines, the number of lines
    """
    img = np.array(img)

    if type(img_wrinkle) == 'NoneType':
        _, img_wrinkle = get_wrinkles(img, resize = resize, is_movie = is_movie, is_red = is_red, background_is_black = False)


    polar_img_wrinkle = polar_coord(img_wrinkle)
    edges = cv2.Canny(polar_img_wrinkle,0,1) 
    edges = cv2.dilate(edges,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    lines = cv2.HoughLinesP(edges,rho=rho,theta=theta, threshold = threshold, minLineLength = int(minLineLength * resize) / 100,
                            maxLineGap = int(maxLineGap * resize) / 100)
    count = 0
    if lines is None:
        return(img, 0, 0, 0)
    else:
        img_return = polar_coord(img)
        median_length = []
        median_dist_center = []
        for line in lines:
            coords = line[0]
            slope = (coords[3]-coords[1])/(coords[2]-coords[0])
            length = sqrt((coords[3] - coords[1])**2 + (coords[2] - coords[0])**2)
            if slope > -0.2 and slope <0.2 and coords[0] < resize / 2 and coords[0] > resize / 4 :
                cv2.line(img_return,(coords[0],coords[1]),(coords[2],coords[3]),[250,250,250],int( resize / 200))
                count += 1
                median_length.append(length)
                median_dist_center.append(min(coords[0], coords[3]))
    img_return = cartesian_coord(img_return)
    median_length = np.median(np.array(median_length))
    median_dist_center = np.median(np.array(median_dist_center))
    return(img_return, count, median_length, median_dist_center)
