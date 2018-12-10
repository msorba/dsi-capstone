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


def threshold_image(img,threshold_val=50):
    """
    
    img : initial image
    threshold_val : the threshold for pixels to keep
    
    returns: 
    img wthat only contains pixel values over "threshold_val".
    This helps tremendously in finding the wrinkles, and allows us to choose smaller n_cluster
    """
    timg = np.asarray(img)
    thresh = timg > threshold_val
    out = thresh * timg
    img = Image.fromarray(out)
    return img


## Newest version of get_classes function. This function takes as input an image, computes kmeans and outputs the labels and the image 'colored' by the classes.
def get_classes(img, resize=128, n_clusters=6, movie=False,threshold_val=100):
    
    """
    img: initial image

    returns: the labels of the pixels and the image colored by the classes
    """
    # convert into image if is array
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        pass
    
    # keeps pixels above threshold
    img = threshold_image(img, threshold_val)
    
    # Resize
    if not movie:
        size=(resize, resize)
        img_resize       = img.resize(size, Image.ANTIALIAS)#resize
        img_array_resize = np.asarray(img_resize)
    else:
        reduce_factor    = 4
        size             = tuple(np.divide(img.shape[:2], reduce_factor).astype(int)) + (3,)
        img_array_resize = np.resize(img, size)


    # Reshape
    w,h,d = tuple(img_array_resize.shape)
    X     = np.reshape(img_array_resize, (w*h,d))

    # Add differences and ratios to red as features
#     a = X[:,0] / (X[:,0] + X[:,2])
#     a[np.isnan(a)] = 0
#     a = np.reshape(a, (resize**2, 1))
#     X = np.append(X, a, axis=1)

#     b = X[:,0] / (X[:,0] + X[:,1])
#     b[np.isnan(b)] = 0
#     b = np.reshape(b, (resize**2, 1))
#     X = np.append(X, b, axis=1)
    
#     c = X[:,0] - X[:,1] + X[:,2]
#     c[np.isnan(c)] = 0
#     c = np.reshape(c, (resize**2, 1))
#     X = np.append(X, c, axis=1)
    
    # Add radial distance as feature
    y_axis = np.array([])
    for i in np.arange(resize):
        y_axis = np.append(y_axis, np.arange(resize))
        
    y_axis          = np.reshape(y_axis, (resize**2, 1))
    x_axis          = np.reshape(np.repeat(np.arange(resize), resize), (resize**2, 1))
    radial_distance = np.sqrt(np.abs(x_axis - (resize / 2))**2 + np.abs(y_axis - (resize / 2))**2)
    X               = np.append(X, radial_distance, axis=1)

    # Remove blue and green
#     X = X[:,[0, 3, 4, 5]]
#     X = X[:,3:]

    
    # Fit model
#     km    = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
#     km = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
    
    # Determine labels and classes
    img_array = np.asarray(img)
    w,h,d     = tuple(img_array.shape)
    labels    = [((X[:,0] > 100) * 1) & np.abs((1 - (X[:,1] / X[:,2])) < 0.75)
                & (X[:,1] < 80) & (X[:,2] < 80) & ((X[:,0]/ (X[:,1] + X[:,2])) > 1.0)] #km.labels_ #km.predict(np.reshape(img_array, (w*h, d)))el
    classes   = np.reshape(labels, size) 
    classes   = np.multiply(classes, 255.0 / np.max(classes)) # Normalize
    
    return(labels,classes)



# This function is depracated and should not be used. 
def get_classes_old(img, resize=128, n_clusters=6, movie=False,threshold_val=100):
    
    """
    img: initial image

    returns: the labels of the pixels and the image colored by the classes
    """
    # convert into image if is array
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        pass
    
    # keeps pixels above threshold
    img = threshold_image(img,threshold_val)
    
    # Resize
    if not movie:
        size=(resize, resize)
        img_resize       = img.resize(size, Image.ANTIALIAS)#resize
        img_array_resize = np.asarray(img_resize)
    else:
        reduce_factor    = 4
        size             = tuple(np.divide(img.shape[:2], reduce_factor).astype(int)) + (3,)
        img_array_resize = np.resize(img, size)

    	
    # Reshape
    w,h,d = tuple(img_array_resize.shape)
    X     = np.reshape(img_array_resize, (w*h,d))

    # Add position variables
    # y_axis = np.array([])
    # for i in np.arange(128):
    # 	y_axis = np.append(y_axis, np.arange(128))

    # y_axis = np.reshape(y_axis, (128 * 128, 1))
    # x_axis = np.reshape(np.repeat(np.arange(128), 128), (128*128, 1))
    # X      = np.append(X, x_axis, axis=1)
    # X      = np.append(X, y_axis, axis=1)

	# Fit model
    km    = KMeans(n_clusters=n_clusters, random_state=0)
    km.fit(X)
    
    # Determine labels and classes
    img_array = np.asarray(img)
    w,h,d     = tuple(img_array.shape)
    labels=km.predict(np.reshape(img_array, (w*h, d)))
    classes   = np.reshape(labels, img_array.shape[:2]) 
    classes   = np.multiply(classes, 255.0 / np.max(classes)) # Normalize
    return(labels,classes)


# This function takes an image as input and outputs the image 'colored' by the wrinkle class.
def get_wrinkle_class(img, resize=128,n_clusters=6, threshold_val=100):
    
    """
    img: initial image

    returns : intial images colorded by the wrinkle class with its label 
    """
	labels    = get_classes(img, resize=resize, n_clusters=n_clusters, movie=False, threshold_val=threshold_val)[0]
	img_array = np.asarray(img)
	img_array = img_array[:,:,:3]
	w,h,d     = tuple(img_array.shape)
	X         = np.reshape(img_array, (w*h,d))
	df        = pd.DataFrame(X, columns=['red', 'green', 'blue'])
	df_mean   = df.astype('int').groupby(labels).mean()

	# Find most red class
# 	df_mean['red_diff'] = (np.divide(df_mean['red'], df_mean['green']) +  
# 	                       np.divide(df_mean['red'], df_mean['blue'])) / 2.0
	df_mean['red_diff'] = df_mean['red'] - df_mean['green'] - df_mean['blue']
	wrinkle_id     = np.argmax(df_mean['red_diff'])

# 	df_mean['most_red'] = (np.abs(df_mean['red'] - 90) +  np.abs(df_mean['red'] - 50) + np.abs(df_mean['blue'] - 50))
# 	wrinkle_id          = np.argmin(df_mean['most_red'])
	wrinkle_labels      = [1 if i == wrinkle_id else 0 for i in labels]

	# Plot
	wrinkle_classes     = np.reshape(wrinkle_labels, img_array.shape[:2]) 
	wrinkle_classes     = np.multiply(wrinkle_classes, 255.0/np.max(wrinkle_classes))

	return(wrinkle_labels,wrinkle_classes)



def detect_lines(img,rho=1,theta=np.pi/180,threshold=150,minLineLength=5,maxLineGap=1):
    """
    input img
    
    returns: the initial images with the lines, the number of lines
    """
    polar_img_wrinkle=polar_coord(get_wrinkle_class(img)[1])
    n_pixels=img.shape[0]+img.shape[1]
    edges = cv2.Canny(polar_img_wrinkle,40,70) 
    edges = cv2.dilate(edges,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    plt.imshow(edges)
    lines = cv2.HoughLinesP(edges,rho=rho,theta=theta, threshold=threshold, minLineLength=int(minLineLength*n_pixels/100),maxLineGap=int(maxLineGap*n_pixels/100))
    count = 0
    if lines is None:
        return(img,0)
    else:
        img_return=polar_coord(img)
        for line in lines:
            coords=line[0]
            slope=(coords[3]-coords[1])/(coords[2]-coords[0])
            if slope > -0.2 and slope <0.2:
                cv2.line(img_return,(coords[0],coords[1]),(coords[2],coords[3]),[250,250,250],int(n_pixels/500))
                count += 1
    img_return = cartesian_coord(img_return)
    return(img_return,count)

def plot_wrinkle_class(img_wrinkle_class, save=True):
    """
    input : wrinkle class img
    
    returns: plots the wrinkle class img and saves it if save = True
    """
    plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.imshow(img_wrinkle_class)

    if save:
        pylab.savefig('static/results/wrinkle.png',bbox_inches='tight')


def perc_wrinkled(img):
    
    """
    input : initial img
    
    returns: the percent of the surface of the biofilm covered by wrinkles
    """
    wrinkle_labels,wrinkle_classes=get_wrinkle_class(img)
    return round(100*sum(np.array(wrinkle_labels)==1)/len(wrinkle_labels),2)


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

