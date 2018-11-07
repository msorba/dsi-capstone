import os, re
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pylab


basepath = "Images"
# basepath = "/home/khalana/GitRepos/Capstone_local/Images"

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
    X: image (array format) having cartesian polarity

    returns: the initial image transformed into polar coordinates 
    """
    r = np.sqrt(((X.shape[0]/2.0)**2.0)+((X.shape[1]/2.0)**2.0)) #get radius
    X_polar = cv2.linearPolar(X,(X.shape[0]/2, X.shape[1]/2),r, cv2.WARP_FILL_OUTLIERS)
    X_polar = X_polar.astype(np.uint8)
    return(X_polar)


# This function transforms polar coordinates to cartesian. X is the image (has to be array)
def cartesian_coord(X):
    r = np.sqrt(((X.shape[0]/2.0)**2.0)+((X.shape[1]/2.0)**2.0)) #get radius
    X_cartesian=cv2.linearPolar(X, (X.shape[0]/2, X.shape[1]/2),r, cv2.WARP_INVERSE_MAP)
    X_cartesian = X_cartesian.astype(np.uint8)
    return(X_cartesian)

def threshold_image(img,threshold_val=50):
    """
    only keeps pixel values over "threshold_val". This helps tremendously in finding the wrinkles, and allows us to choose smaller n_cluster
    """
    timg = np.asarray(img)
    thresh = timg > threshold_val
    out = thresh * timg
    img = Image.fromarray(out)
    return img


# This function takes as input an image, computes kmeans and outputs the labels and the image 'colored' by the classes.
def get_classes(img, resize=128, n_clusters=3, movie=False,threshold_val=50):
    
    """
    img: initial image

    returns: the labels of the pixels and the image colored by the classes
    """
    # convert into image if is array
    try:
        img=Image.fromarray(img, 'RGB')
    except:
        pass
    
    # keeps pixels above threshold
    img=threshold_image(img,threshold_val)
    
    # Resize
    if not movie:
        size=(resize,resize)
        img_resize       = img.resize(size, Image.ANTIALIAS)#resize
        img_array_resize = np.asarray(img_resize)
    else:
        reduce_factor    = 4
        size             = tuple(np.divide(img.shape[:2], reduce_factor).astype(int)) + (3,)
        img_array_resize = np.resize(img, size)

    	
    # Reshape
    w,h,d = tuple(img_array_resize.shape)
    X  = np.reshape(img_array_resize, (w*h,d))
    km = KMeans(n_clusters=n_clusters, random_state=0)
    km.fit(X)
    
    img_array=np.asarray(img)
    w,h,d = tuple(img_array.shape)
    labels=km.predict(np.reshape(img_array, (w*h, d)))
    classes = np.reshape(labels, img_array.shape[:2]) 
    classes = np.multiply(classes, 255.0/np.max(classes)) # Normalize
    return(labels,classes)


# This function takes an image as input and outputs the image 'colored' by the wrinkle class.
def get_wrinkle_class(img,resize=128,n_clusters=3,threshold_val=50):
    labels=get_classes(img,resize=resize,n_clusters=n_clusters,threshold_val=140)[0]
    img_array=np.asarray(img)
    img_array = img_array[:,:,:3]
    w,h,d = tuple(img_array.shape)
    X  = np.reshape(img_array, (w*h,d))
    df = pd.DataFrame(X, columns=['red', 'green', 'blue'])
    df_mean = df.astype('int').groupby(labels).mean()

    # Find most red class
#     df_mean['red_diff'] = (np.divide(df_mean['red'], df_mean['green']) +  
#                            np.divide(df_mean['red'], df_mean['blue'])) / 2.0
    df_mean['red_diff'] = df_mean['red'] - df_mean['green'] - df_mean['blue']
    wrinkle_id     = np.argmax(df_mean['red_diff'])
    wrinkle_labels = [1 if i == wrinkle_id else 0 for i in labels]
    wrinkle_classes = np.reshape(wrinkle_labels, img_array.shape[:2]) 
    wrinkle_classes = np.multiply(wrinkle_classes, 255.0/np.max(wrinkle_classes))
    return(wrinkle_labels,wrinkle_classes)



def detect_lines(img,rho=1,theta=np.pi/180,threshold=150,minLineLength=200,maxLineGap=30):
    """
    input img
    
    returns: the initial images with the lines, the number of lines
    """
    polar_img_wrinkle=polar_coord(get_wrinkle_class(img)[1])
    edges = cv2.Canny(polar_img_wrinkle,40,70) 
    lines = cv2.HoughLinesP(edges,rho=rho,theta=theta, threshold=threshold, minLineLength=minLineLength,maxLineGap=maxLineGap)
    count = 0
    if lines is None:
        return(img,0)
    else:
        img_return=polar_coord(img)
        for line in lines:
            coords=line[0]
            slope=(coords[3]-coords[1])/(coords[2]-coords[0])
            if slope > -0.2 and slope <0.2:
                cv2.line(img_return,(coords[0],coords[1]),(coords[2],coords[3]),[255,255,255],3)
                count += 1
    img_return=cartesian_coord(img_return)
    return(img_return,count)

def plot_wrinkle_class(img_wrinkle_class, save=True):
    plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.imshow(img_wrinkle_class)

    if save:
        pylab.savefig('static/results/wrinkle.png',bbox_inches='tight')




