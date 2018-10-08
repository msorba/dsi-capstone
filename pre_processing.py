import PIL
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re

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
                img = PIL.Image.open(mypath)
                img = img.resize(size)
                arr = np.array(img).astype('uint8')
                images.append(arr)
    else:
        for file in file_list_or_directory:
            img = PIL.Image.open(file)
            img = img.resize(size)
            arr = np.array(img).astype('uint8')
            images.append(arr)
    return images