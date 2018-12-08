import os, re
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pylab


basepath = "Images"

def get_data_from_filenames(basepath):
    """
    uses file structure and filename to extract relevant information.
    basepath:: the relative directory to scan for images. basepath must contain a (sub-)directory called 'Images'
    returns: dataframe containing information from filenames
    """
    full_data = []
    for root, dirs, files in os.walk(basepath, topdown=True):
        for name in files:
            data = name.split(".")[0]
            data = data.split("_")
            method = root.split("/")[root.split("/").index("Images") + 1]
            data.insert(0,method)            #add method from directory name to data
            if method == 'Scanner':
                if len(data) < 5:            # make all data consistent length
                    data.append("-")
                if data[4].isdigit():
                    data.insert(4,"-")       # some filenames include background color (W or B). This accounts for missing values.
                if len(data) == 5:
                    data.append('0')         # some filepaths have "_1" or "_2" at the end. This adds "0" for missing values. 
            data[3] = int(data[3][1])

            if method == 'Keyence':          # images in Keyence directory had simpler format. Adding NAs for data consistency,
                data.append("NA")
                data.append("NA")

            data.append(os.path.join(root,name))
            full_data.append(data)
    return pd.DataFrame(full_data,columns=['method','genotype','date','day','background_color','img_num??','filepath'])


def load_images(file_list_or_directory,size=(1000,1000)):
    """
    file_list_or_directory:: (string, list, pd.core.series)::
                            you can give a list of full filepaths (using 'filepath' column from get data function dataframe)
                            OR a string to specify diectory
    returns: array of image arrays
    """
    images = []
    
    if type(file_list_or_directory) == str:
        for root, dirs, files in os.walk(file_list_or_directory, topdown=True):
            for name in files:
                mypath = os.path.join(root,name)
                img = Image.open(mypath)
                img = img.resize(size)
                arr = np.array(img).astype('uint8')
                images.append(arr)
    else:
        for file in file_list_or_directory:
            img = Image.open(file)
            img = img.resize(size)
            arr = np.array(img).astype('uint8')
            images.append(arr)
    return images


# This function transforms cartesian coordinates to polar. X is the image (has to be array)

def polar_coord(X):
    """
    X: image (array format) having cartesian coordinates

    returns: the initial image transformed into polar coordinates 
    """
    r = np.sqrt(((X.shape[0]/2.0)**2.0)+((X.shape[1]/2.0)**2.0)) #get radius
    X_polar = cv2.linearPolar(X,(X.shape[0]/2, X.shape[1]/2),r, cv2.WARP_FILL_OUTLIERS)
    X_polar = X_polar.astype(np.uint8)
    return(X_polar)


# This function transforms polar coordinates to cartesian. X is the image (has to be array)
def cartesian_coord(X):
    """
    X: image (array format) 

    returns: the initial image transformed into cartesian coordinates 
    """
    r = np.sqrt(((X.shape[0]/2.0)**2.0)+((X.shape[1]/2.0)**2.0)) #get radius
    X_cartesian=cv2.linearPolar(X, (X.shape[0]/2, X.shape[1]/2),r, cv2.WARP_INVERSE_MAP)
    X_cartesian = X_cartesian.astype(np.uint8)
    return(X_cartesian)


def get_wrinkles(img, resize = 128, movie = False, background_is_black = True):
    
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
    if not movie:
        size = (resize, resize)
        img_resize       = img.resize(size, Image.ANTIALIAS)#resize
        img_array_resize = np.asarray(img_resize)
    else:
        reduce_factor    = 4
        size             = tuple(np.divide(img.shape[:2], reduce_factor).astype(int)) + (3,)
        img_array_resize = np.resize(img, size)


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
    lower_black = np.array([0,0,0], dtype = "uint16")
    upper_black = np.array([40,40,40], dtype = "uint16")
    black_mask = cv2.inRange(img_array_resize, lower_black, upper_black)
    classes   = np.reshape(labels, size)
    
    if background_is_black == True :
        classes[black_mask > 0] = -1
    return(labels, classes)


def perc_wrinkled(img, background_is_black = True, resize = 128):
    
    """
    input : initial img
    
    returns: the percent of the surface of the biofilm covered by wrinkles
    """
    wrinkle_labels, _ = get_wrinkles(img, background_is_black = background_is_black, resize = resize)
    return round(100*sum(np.array(wrinkle_labels) == 1) / sum(np.array(wrinkle_labels) == 0), 2)



def detect_spokes(img, rho = 1, theta = np.pi/180, minLineLength = 3, maxLineGap = 1, resize = 128, threshold = 10):
    """
    input img
    
    returns: the initial images with the lines, the number of lines
    """
    _, img_wrinkle = get_wrinkles(img, resize = resize, background_is_black = False)
    polar_img_wrinkle = polar_coord(img_wrinkle)
    edges = cv2.Canny(polar_img_wrinkle,0,1) 
    edges = cv2.dilate(edges,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    lines = cv2.HoughLinesP(edges,rho=rho,theta=theta, threshold = threshold, minLineLength = 
                            int((minLineLength * resize) / 50),maxLineGap=int((maxLineGap * resize) / 50))
    count = 0
    if lines is None:
        return(img,0)
    else:
        img_return = polar_coord(img)
        for line in lines:
            coords = line[0]
            slope = (coords[3]-coords[1])/(coords[2]-coords[0])
            if slope > -0.3 and slope <0.3:
                cv2.line(img_return,(coords[0],coords[1]),(coords[2],coords[3]),[250,250,250],int(resize / 300))
                count += 1
    img_return = cartesian_coord(img_return)
    return(img_return, count)


def plot_wrinkle_class(img_wrinkle_class, save=True):
    """
    input : wrinkle class img
    
    returns: plots the wrinkle class img and saves it if save = True
    """
    plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.imshow(img_wrinkle_class)

    if save:
        pylab.savefig('static/results/wrinkle.png',bbox_inches='tight')



def detect_lines_cy(img, rho=1, theta=np.pi / 180, threshold=150, minLineLength=5, maxLineGap=1):
    """
    input img

    returns: image with lines and lines
    """
    polar_img_wrinkle = polar_coord(get_wrinkle_class(img)[1])
    n_pixels = img.shape[0] + img.shape[1]
    edges = cv2.Canny(polar_img_wrinkle, 40, 70)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    lines = cv2.HoughLinesP(edges, rho=rho, theta=theta, threshold=threshold,
                            minLineLength=int(minLineLength * n_pixels / 100),
                            maxLineGap=int(maxLineGap * n_pixels / 100))
    pol_lines = []
    if lines is None:
        return None, []
    else:
        img_return = polar_coord(img)
        for line in lines:
            coords = line[0]
            slope = (coords[3] - coords[1]) / (coords[2] - coords[0])
            if slope > -0.2 and slope < 0.2:
                cv2.line(img_return, (coords[0], coords[1]), (coords[2], coords[3]), [250, 250, 250],
                         int(n_pixels / 500))
                pol_lines.append(line)
    img_return = cartesian_coord(img_return)
    return (img_return, pol_lines)

