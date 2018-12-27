# Exploring the Structure in Images of Biofilms
## Capstone Project - Columbia University Data Science Institute
#### Contributors: Marianne Sorba, Isaac Wainstein, Chuqi Yang, Ramy Jaber
#### Biology Lab Sponsor: GSAS - Professor Lars Dietrich, Lisa Kahl, Hannah Dayton

## Project Description
A biofilm is a community of bacteria encased in a scaffold that protects the microorganisms. Biofilm structures vary depending on the types of bacteria, the bacterial species and their genetic makeups, as well as external conditions. This project studies images of biofilms and attempts to quantify their structures and distinct morphologies. Quantifying the biofilm morphology has implications in diagnosing and treating bacterial infections.

In this Capstone project, a team of Data Science Masters students explore ways to quantify these structures using image processing and supervised and unsupervised machine learning. 

## In this Repo
- Jupyter Notebooks to walk through the analysis completed. 
- Copy of Flask Application - live version available at <href "dsicapstonegsas.pythonanywhere.com" />
- archive: work performed throughout the project that did not make it into the final submissions

## Image Processing Tutorial
This tutorial takes the user through the process of loading a single image or a directory of many images and applying the functions to detect features. 
- Percent Wrinkle
- Detecting spokes
- Measuring spoke size
- Dimensionality reduction
- Data representation and visualization

## Video Processing Tutorial
Similar to the image tutorial, this notebook follows the process to load a video, split it into frames, process the frames individually, and save a new video
Note that cropping a video requires some manual tweaking within the notebook to find the correct bounding box for each biofilm in the video. 

## Classification Tutorial
This notebook uses the entire dataset of day 4 and day 5 images to train an SVM classifier. The training image set is augmented through rotation and transposition. Then the training data is fit to a PCA transformation with 50 Principle Components. Finally, the classifier is fit using an RBF kernel and GridSearch parameter tuning. Current accuracy on an original data set of 512 images (augmented to ~4000) is approx. 90% classification accuracy (F-1 score)


## Flask App
- appy.py
- app_functions.py
- image_processing_functions.py
- templates/*
- static/*

These files/directories are all necessary for the Flask App to run. These files are downloaded from pythonAnywhere.com which is where the app is curently hosted. 

Note to future developers: When running locally, you will likely have to debug some file path issues since the path structure in pythonAnywhere is non-standard. 
to run locally: 
1. update the "app.secret_key" variable in app.py
2. In this directory, open a terminal and run:

$export FLASK_APP=app.py

$flask run

3. go to 127.0.0.1:5000 in your browser

questions about the Flask App can be sent to Ramy Jaber at rij2105@columbia.edu
