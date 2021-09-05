# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 04:27:43 2021

@author: Bhargav 
"""

import streamlit as st 
import altair as alt 
import urllib
import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.model_selection import train_test_split 
import pdb
from tensorflow import keras 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, BatchNormalization, Dropout, Input, MaxPool2D , Flatten
import cv2
import imgaug.augmenters as iaa
from PIL import Image, ImageDraw
from PIL import ImagePath
from tensorflow.keras.models import Model, load_model
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import LearningRateScheduler
import random
from tqdm import tqdm
from sklearn.metrics import roc_curve,precision_recall_curve, auc , confusion_matrix
import seaborn as sns
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB7
import tensorflow as tf
import segmentation_models as sm
from segmentation_models.metrics import iou_score
from numpy import save ,load
from pathlib import Path

class Unet_efficnetB7 :
  def __init__(self,input_shape,classes):
    tf.keras.backend.clear_session()
    self.input_shape = input_shape
    self.classes = classes
    self.inputs = Input(input_shape)
    self.encoder = EfficientNetB7(include_top=False, weights='imagenet', input_tensor= self.inputs)

  def conv_block(self,inputs,num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

  def decoder_block(self,inputs,skip, num_filters):
      x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
      x = Concatenate()([x, skip])
      x = self.conv_block(x, num_filters)
      return x

  def build_efficient_unet(self):
    
    s1 = self.encoder.get_layer("input_1").output  #skip connection 256X1600 

    s2 = self.encoder.get_layer("block2a_expand_activation").output  #skip connection 128X800

    s3 = self.encoder.get_layer("block3a_expand_activation").output  #skip connection 64X400

    s4 = self.encoder.get_layer("block4a_expand_activation").output  #skip connection 32X200

    s5 = self.encoder.get_layer("block6a_expand_activation").output  #skip connection 16X100

    """ Bottle neck"""

    b1 = self.encoder.get_layer("top_activation").output  # 8X50

    """decoder block"""
    d1 = self.decoder_block(b1,s5,512) #16 X 100 X 512 

    d2 = self.decoder_block(d1,s4,256) #32 X 200 X 256

    d3 = self.decoder_block(d2,s3,128) #64 X 400 X 128
  
    d4 = self.decoder_block(d3,s2,64) #128 X 800 X 64

    d5 = self.decoder_block(d4,s1,32) #256 X 1600 X 32

    """Output"""
    outputs = Conv2D(self.classes,1,padding = "same" , activation= "softmax")(d5) 
    
    unet_model = Model(self.inputs , outputs, name = "EfficientNetB4_UNET") 

    return  unet_model



current_cwd = os.getcwd()
train_images_path = os.path.join(current_cwd,'train_images')

# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(markdown_file):
    return Path(markdown_file).read_text()

def main() :
    #Render the readme markdown using st.markdown 
    readme_text = st.markdown(get_file_content_as_string("instructions.md"))
    
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",["Show instructions","Run the app","Show the source code"])
    
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app')
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string("streamlit_app.py"))
    elif app_mode == "Run the app" :
        readme_text.empty()
        run_the_app()
    
#This is the main app itself, which appears when the user selects on the run_the_app     
def run_the_app():
    # to make Streamlit fast st.cache allows us to reuse computation accross runs 
    # in this common pattern, we download the data only once 
    @st.cache
    def load_metadata(csv_path):
        return pd.read_csv(csv_path)
    
    metadata = load_metadata('train.csv')
    X_train = load_metadata('X_train.csv')
    X_test = load_metadata('X_test.csv')   
    st.write('## **Metadata**', metadata[:1000])
    st.write('## **Summary**' , X_train[:1000])
    
    st.header('Exploratory Data Analysis') 
          
    st.subheader('1- Distribution of steel images among defect and non defect')
    
    #Images with defect and non defect steel
    n_total_images = X_train.shape[0]
    n_defect_images = sum(X_train.hasDefect) 
    n_nodefect_images = n_total_images -  n_defect_images
    
    plt.figure(figsize = (3,2)) 
    plt.xlabel('Type of image')
    plt.ylabel('Percentage')
    plt.bar(['Defect steel', 'Non Defect steel'], [round(((n_defect_images/ n_total_images)* 100 ),2), round(((n_nodefect_images/ n_total_images)* 100 ),2)],
            color = ['orange','blue'] , width = 0.8)
    plt.title('Distribution of steel among defect and non defect')
    plt.show()
    st.pyplot(plt)
    
    st.write('There are ',n_total_images,' Total images')
    st.write('There are ',n_defect_images,' with atleast 1 defect images')
    st.write('There are ', n_nodefect_images, ' 0 defect images')
    st.write(round(((n_defect_images/ n_total_images)* 100 ),2),'% of defect steel images and ',round(((n_nodefect_images/ n_total_images)* 100 ),2),' % of Non defect images')
 
    
    st.subheader('2- Distribution of defect types')
    plt.figure(figsize = (3,2)) 
    plt.xlabel('Defect type')
    plt.ylabel('Number')
    plt.bar([1,2,3,4],X_train[['hasDefect_1','hasDefect_2','hasDefect_3','hasDefect_4']].sum(axis = 0),
            color = ['red','green','blue','orange'] , width = 0.8)
    xlocs, xlabs = plt.xticks()
    plt.title('Distribution of steel among defect types')
    for i, v in zip([1,2,3,4],X_train[['hasDefect_1','hasDefect_2','hasDefect_3','hasDefect_4']].sum(axis = 0)):
        plt.text(xlocs[i] - 0.15, v + 0.5, str(v))
    plt.show()
    st.pyplot(plt)
    st.write('**Observation:** There are 4127 type 3 defects and just 198 type 2 defects. Hence the data is highy imbalanced.')

    st.subheader('3- Distribution of number of defects in each image')
    plt.figure(figsize = (3,2)) 
    plt.xlabel('number of defects in each image')
    plt.ylabel('Number')
    plt.bar([str(0),str(1),str(2),str(3)], X_train[['hasDefect_1','hasDefect_2','hasDefect_3','hasDefect_4']].sum(axis = 1).value_counts().sort_index(),
            color = ['orange','blue','green','red'] , width = 0.8)
    plt.title('Distribution of number of defects in each image')
    
    xlocs, xlabs = plt.xticks()
    for i, v in zip( [0,1,2,3], X_train[['hasDefect_1','hasDefect_2','hasDefect_3','hasDefect_4']].sum(axis = 1).value_counts().sort_index()):
        plt.text(xlocs[i] - 0.15, v + 0.5, str(v))
    plt.show()    
    st.pyplot(plt)
    st.write('**Observation:** There are just 345 images with multi defects.')
    
    st.subheader('4- Images with no defects')
    
    tmp = []
    cnt=0
    print("Sample images with no defects:")
    for i in X_train['ImageId'][X_train['hasDefect']==0]:
        if cnt<3:
            fig, ax = plt.subplots(1,1,figsize=(8, 7))
            # img = Image.open(os.path.join(train_images_path,i))
            img = cv2.imread(os.path.join(train_images_path , i))
            plt.imshow(img)
            ax.set_title(i)
            plt.show()
            st.pyplot(fig)
            cnt+=1
            
    st.write('**Observation:** We can observe some marks on non defect images aswell. However, our task is to recognize particular type of 4 defects')
    
    colourmap = [[0, 0, 0], [255, 105, 180], [ 180,255,105],[ 105, 180,255], [255, 255,105]]
    classes_tocolour =   dict({0: [0, 0, 0], 1: [255, 105, 180], 2:  [180,255,105], 3:[105, 180,255], 4: [ 255, 255,105]})
    classes = [0,1,2,3,4] 
    
    @st.cache
    def rle2_1frame(mask_rle_list, shape=(1600,256)):
        '''
        mask_rle: list of strings(run-length as string formated (start length)), each for 1,2,3,4 defects 
        shape: (width,height) of array to return 
        Returns 2 D numpy array. 0 for no defect, 1 defect 1 , 2 for defect 2 , 3 for defect 3 and 4 for defect 4
        This function is specific to this competition
        '''
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)    
        for i in range(len(mask_rle_list)):
          if mask_rle_list[i] != ' ' :
            sp = mask_rle_list[i].split()
            starts, lengths = [np.asarray(x, dtype=int) for x in (sp[0:][::2], sp[1:][::2])]
            starts -= 1
            ends = starts + lengths
            # pdb.set_trace()
            for lo, hi in zip(starts, ends):
                img[lo:hi] = i+1 
        img = img.reshape(shape).T      
        return img #256 * 1600
    
    @st.cache
    def rle_to_RGBmask(mask_rle_list, classes_tocolour ,shape=(1600,256)):
      ''' 
      This function will save the RGB masks from RLE
      '''
      img = rle2_1frame(mask_rle_list, shape=(1600,256))
      # pdb.set_trace()
      #Till here we got 256 X 1600 and now we have to convert this matrix to an RGB encoded mask
      RGB_image = []
      for outer in img :
        col = []
        for inner in outer :
          col.append(classes_tocolour.get(inner))  
        RGB_image.append(col)
      return np.array(RGB_image) #256 X 1600 X 3 
    
    @st.cache
    def RGBmask_to_width_height_classes ( rgb_mask , colourmap ):
      ''' 
      This function will convert the RGB mask to width X height X classes
      '''
      output_mask = []
      for i , color in enumerate(colourmap): 
        cmap = np.all(np.equal(rgb_mask , color ), axis = -1)
        cmap.astype(int)  
        output_mask.append(cmap)
      output_mask = np.stack(output_mask , axis = -1)
      output_mask = output_mask.astype(np.uint8)
      return output_mask #output will have five channels. any pixel will have 1 in any one of the 5 channels 
    
    @st.cache
    def width_height_classes_toRGB(img,classes_tocolour):
      ''' Given an widthXheightXclasses we need to convert into an RGB image 256X1600X5  to 256 X 1600 X 3'''
      img = np.argmax(img,axis= -1) #256 X 1600
      RGB_image = []
      for outer in img :
        col = []
        for inner in outer :
          col.append(classes_tocolour.get(inner))  
        RGB_image.append(col)
      return np.array(RGB_image) #256 X 1600 X 3 
    
    @st.cache
    def one_frame_rgb(img,classes_tocolour):
      RGB_image = []
      for outer in img :
        col = []
        for inner in outer :
          col.append(classes_tocolour.get(inner))  
        RGB_image.append(col)
      return np.array(RGB_image) #256 X 1600 X 3 
  
    # Visualization: Sample images having No defects
    st.subheader('5.Images and masks with out any defects')
    cnt = 0
    for index,row in X_train[X_train.hasDefect == 0].iterrows(): 
      if cnt < 3:
        fig, (ax1,ax2) = plt.subplots(nrows = 1,ncols = 2,figsize=(15, 10))
        img = cv2.imread(os.path.join(train_images_path , row['ImageId']))
        ax1.imshow(img)
        ax1.set_title(row['ImageId']) 
        
        file_name = (row['ImageId']).split('.')[0] 
        
        mask_rle_list = [row.Defect_1,row.Defect_2,row.Defect_3,row.Defect_4] 
        rgb_mask = rle_to_RGBmask(mask_rle_list, classes_tocolour, shape=(1600, 256))
        original_mask = RGBmask_to_width_height_classes(rgb_mask, colourmap)  #256 X 1600 X 5         
        
        original_mask = width_height_classes_toRGB(original_mask, classes_tocolour)
        # img = cv2.imread(os.path.join(train_images_path , i))
        ax2.imshow(original_mask)
        ax2.set_title(file_name + '_RGB_mask')    
        plt.show()
        st.pyplot(plt)
        cnt+=1 
      else :
        break  
    st.write('**Note**: The non defect pixels are represeted with black colour')
    
# Visualization: Sample images having defect 1
    st.subheader('6. Images and masks with type1 defect')
    cnt = 0
    for index,row in X_train[(X_train.hasDefect_1 == 1) & (X_train.hasDefect_2 == 0) & (X_train.hasDefect_3 == 0) & (X_train.hasDefect_4 == 0)].iterrows(): 
      if cnt < 3:
        fig, (ax1,ax2) = plt.subplots(nrows = 1,ncols = 2,figsize=(15, 10))
        img = cv2.imread(os.path.join(train_images_path , row['ImageId']))
        ax1.imshow(img)
        ax1.set_title(row['ImageId']) 
        
        file_name = (row['ImageId']).split('.')[0] 
        
        mask_rle_list = [row.Defect_1,row.Defect_2,row.Defect_3,row.Defect_4] 
        rgb_mask = rle_to_RGBmask(mask_rle_list, classes_tocolour, shape=(1600, 256))
        original_mask = RGBmask_to_width_height_classes(rgb_mask, colourmap)  #256 X 1600 X 5         
        
        original_mask = width_height_classes_toRGB(original_mask, classes_tocolour)
        # img = cv2.imread(os.path.join(train_images_path , i))
        ax2.imshow(original_mask)
        ax2.set_title(file_name + '_RGB_mask')    
        plt.show()
        st.pyplot(plt)
        cnt+=1 
      else :
        break
    st.write('**Note**: The defect 1 type pixels are represeted with hot pink colour')
    
# Visualization: Sample images having defect 2
    st.subheader('7. Images and masks with type2 defect')
    cnt = 0
    for index,row in X_train[(X_train.hasDefect_1 == 0) & (X_train.hasDefect_2 == 1) & (X_train.hasDefect_3 == 0) & (X_train.hasDefect_4 == 0)].iterrows(): 
      if cnt < 3:
        fig, (ax1,ax2) = plt.subplots(nrows = 1,ncols = 2,figsize=(15, 10))
        img = cv2.imread(os.path.join(train_images_path , row['ImageId']))
        ax1.imshow(img)
        ax1.set_title(row['ImageId']) 
        
        file_name = (row['ImageId']).split('.')[0] 
        
        mask_rle_list = [row.Defect_1,row.Defect_2,row.Defect_3,row.Defect_4] 
        rgb_mask = rle_to_RGBmask(mask_rle_list, classes_tocolour, shape=(1600, 256))
        original_mask = RGBmask_to_width_height_classes(rgb_mask, colourmap)  #256 X 1600 X 5         
        
        original_mask = width_height_classes_toRGB(original_mask, classes_tocolour)
        # img = cv2.imread(os.path.join(train_images_path , i))
        ax2.imshow(original_mask)
        ax2.set_title(file_name + '_RGB_mask')    
        plt.show()
        st.pyplot(plt)
        cnt+=1 
      else :
        break
    st.write('**Note**: The defect 2 type pixels are represeted with hot green colour')
    
# Visualization: Sample images having defect 3
    st.subheader('8. Images and masks with type3 defect')
    cnt = 0
    for index,row in X_train[(X_train.hasDefect_1 == 0) & (X_train.hasDefect_2 == 0) & (X_train.hasDefect_3 == 1) & (X_train.hasDefect_4 == 0)].iterrows(): 
      if cnt < 3:
        fig, (ax1,ax2) = plt.subplots(nrows = 1,ncols = 2,figsize=(15, 10))
        img = cv2.imread(os.path.join(train_images_path , row['ImageId']))
        ax1.imshow(img)
        ax1.set_title(row['ImageId']) 
        
        file_name = (row['ImageId']).split('.')[0] 
        
        mask_rle_list = [row.Defect_1,row.Defect_2,row.Defect_3,row.Defect_4] 
        rgb_mask = rle_to_RGBmask(mask_rle_list, classes_tocolour, shape=(1600, 256))
        original_mask = RGBmask_to_width_height_classes(rgb_mask, colourmap)  #256 X 1600 X 5         
        
        original_mask = width_height_classes_toRGB(original_mask, classes_tocolour)
        # img = cv2.imread(os.path.join(train_images_path , i))
        ax2.imshow(original_mask)
        ax2.set_title(file_name + '_RGB_mask')    
        plt.show()
        st.pyplot(plt)
        cnt+=1 
      else :
        break
    st.write('**Note**: The defect 3 type pixels are represeted with hot blue colour')
    
# Visualization: Sample images having defect 4
    st.subheader('9. Images and masks with type4 defect')
    cnt = 0
    for index,row in X_train[(X_train.hasDefect_1 == 0) & (X_train.hasDefect_2 == 0) & (X_train.hasDefect_3 == 0) & (X_train.hasDefect_4 == 1)].iterrows(): 
      if cnt < 3:
        fig, (ax1,ax2) = plt.subplots(nrows = 1,ncols = 2,figsize=(15, 10))
        img = cv2.imread(os.path.join(train_images_path , row['ImageId']))
        ax1.imshow(img)
        ax1.set_title(row['ImageId']) 
        
        file_name = (row['ImageId']).split('.')[0] 
        
        mask_rle_list = [row.Defect_1,row.Defect_2,row.Defect_3,row.Defect_4] 
        rgb_mask = rle_to_RGBmask(mask_rle_list, classes_tocolour, shape=(1600, 256))
        original_mask = RGBmask_to_width_height_classes(rgb_mask, colourmap)  #256 X 1600 X 5         
        
        original_mask = width_height_classes_toRGB(original_mask, classes_tocolour)
        # img = cv2.imread(os.path.join(train_images_path , i))
        ax2.imshow(original_mask)
        ax2.set_title(file_name + '_RGB_mask')    
        plt.show()
        st.pyplot(plt)
        cnt+=1 
      else :
        break
    st.write('**Note**: The defect 4 type pixels are represeted with yellow colour')   
    
# Visualization: Sample images having defect 1, 2 and 3
    st.subheader('10. Images and masks with type 1, 2 and 3 defects')
    cnt = 0
    for index,row in X_train[(X_train.hasDefect_1 == 1) & (X_train.hasDefect_2 == 1) & (X_train.hasDefect_3 == 1) & (X_train.hasDefect_4 == 0)].iterrows(): 
      if cnt < 3:
        fig, (ax1,ax2) = plt.subplots(nrows = 1,ncols = 2,figsize=(15, 10))
        img = cv2.imread(os.path.join(train_images_path , row['ImageId']))
        ax1.imshow(img)
        ax1.set_title(row['ImageId']) 
        
        file_name = (row['ImageId']).split('.')[0] 
        
        mask_rle_list = [row.Defect_1,row.Defect_2,row.Defect_3,row.Defect_4] 
        rgb_mask = rle_to_RGBmask(mask_rle_list, classes_tocolour, shape=(1600, 256))
        original_mask = RGBmask_to_width_height_classes(rgb_mask, colourmap)  #256 X 1600 X 5         
        
        original_mask = width_height_classes_toRGB(original_mask, classes_tocolour)
        # img = cv2.imread(os.path.join(train_images_path , i))
        ax2.imshow(original_mask)
        ax2.set_title(file_name + '_RGB_mask')    
        plt.show()
        st.pyplot(plt)
        cnt+=1 
      else :
        break 
    
# Visualization: Sample images having defect 3 and 4
    st.subheader('11. Images and masks with type 3 and 4 defects')
    cnt = 0
    for index,row in X_train[(X_train.hasDefect_1 == 0) & (X_train.hasDefect_2 == 0) & (X_train.hasDefect_3 == 1) & (X_train.hasDefect_4 == 1)].iterrows(): 
      if cnt < 3:
        fig, (ax1,ax2) = plt.subplots(nrows = 1,ncols = 2,figsize=(15, 10))
        img = cv2.imread(os.path.join(train_images_path , row['ImageId']))
        ax1.imshow(img)
        ax1.set_title(row['ImageId']) 
        
        file_name = (row['ImageId']).split('.')[0] 
        
        mask_rle_list = [row.Defect_1,row.Defect_2,row.Defect_3,row.Defect_4] 
        rgb_mask = rle_to_RGBmask(mask_rle_list, classes_tocolour, shape=(1600, 256))
        original_mask = RGBmask_to_width_height_classes(rgb_mask, colourmap)  #256 X 1600 X 5         
        
        original_mask = width_height_classes_toRGB(original_mask, classes_tocolour)
        # img = cv2.imread(os.path.join(train_images_path , i))
        ax2.imshow(original_mask)
        ax2.set_title(file_name + '_RGB_mask')    
        plt.show()
        st.pyplot(plt)
        cnt+=1 
      else :
        break
    
    st.header('Predicting the defects using Unet model with efficient net as backbone')    
    input_shape  = (256,1600,3)
    ueff = Unet_efficnetB7(input_shape = input_shape,classes = 5 )
    unet_model = ueff.build_efficient_unet()
    unet_model.load_weights('best_multi_class_model.h5')
    
# # Visualization: Sample original and predicted masks
    st.subheader('1. Images, masks and predicted masks with out any defects')
    cnt = 0
    for index,row in X_test[X_test.hasDefect == 0 ].iterrows() :
      if cnt < 3:
        fig, (ax1,ax2,ax3) = plt.subplots(nrows = 1,ncols = 3,figsize=(25, 6))
        img = cv2.imread(os.path.join(train_images_path,row['ImageId']))
        ax1.imshow(img)
        ax1.set_title('Original_image_'+row['ImageId'])
        
        file_name = (row['ImageId']).split('.')[0] 
        
        mask_rle_list = [row.Defect_1,row.Defect_2,row.Defect_3,row.Defect_4] 
        rgb_mask = rle_to_RGBmask(mask_rle_list, classes_tocolour, shape=(1600, 256))
        original_mask = RGBmask_to_width_height_classes(rgb_mask, colourmap)  #256 X 1600 X 5
        
        original_mask = width_height_classes_toRGB(original_mask, classes_tocolour)
        ax2.imshow(original_mask)
        ax2.set_title('Original_mask_'+file_name)
        
        img = cv2.imread(os.path.join(train_images_path,row['ImageId']))
        predict_mask = unet_model.predict(np.expand_dims(img, axis=0)).argmax(axis = -1) #256X1600X5 
        predict_mask  = np.squeeze(predict_mask, axis=0)
        predict_mask  = one_frame_rgb(predict_mask,classes_tocolour)
        
        # predict_mask = width_height_classes_toRGB(predict_mask, classes_tocolour)
        ax3.imshow(predict_mask)
        ax3.set_title('Predicted Mask_'+os.path.split(i)[1])
        plt.show()
        st.pyplot(plt)
        cnt+=1 
      else :
        break  
    st.write('Images with no defects are being correctly idetified by the model')
    
## Visualization: Sample original and predicted masks
    st.subheader('2. Images, masks and predicted masks with type1 defects')
    cnt = 0
    for index,row in X_test[(X_test.hasDefect_1 == 1) & (X_test.hasDefect_2 == 0) & (X_test.hasDefect_3 == 0) & (X_test.hasDefect_4 == 0)].iterrows() :
      if cnt < 3:
        fig, (ax1,ax2,ax3) = plt.subplots(nrows = 1,ncols = 3,figsize=(25, 6))
        img = cv2.imread(os.path.join(train_images_path,row['ImageId']))
        # pdb.set_trace()
        ax1.imshow(img)
        ax1.set_title('Original_image_'+row['ImageId'])
        
        file_name = (row['ImageId']).split('.')[0] 
        
        mask_rle_list = [row.Defect_1,row.Defect_2,row.Defect_3,row.Defect_4] 
        rgb_mask = rle_to_RGBmask(mask_rle_list, classes_tocolour, shape=(1600, 256))
        original_mask = RGBmask_to_width_height_classes(rgb_mask, colourmap)  #256 X 1600 X 5
        
        original_mask = width_height_classes_toRGB(original_mask, classes_tocolour)
        ax2.imshow(original_mask)
        ax2.set_title('Original_mask_'+file_name)
        
        img = cv2.imread(os.path.join(train_images_path,row['ImageId']))
        predict_mask = unet_model.predict(np.expand_dims(img, axis=0)).argmax(axis = -1) #256X1600X5 
        # pdb.set_trace() 
        predict_mask  = np.squeeze(predict_mask, axis=0)
        predict_mask  = one_frame_rgb(predict_mask,classes_tocolour)
        
        # predict_mask = width_height_classes_toRGB(predict_mask, classes_tocolour)
        ax3.imshow(predict_mask)
        ax3.set_title('Predicted Mask_'+os.path.split(i)[1])
        plt.show()
        st.pyplot(plt)
        cnt+=1 
      else :
        break  
    st.write('Most of the pixels with type1 defects are beig correctly idetified by the model')    
    
## Visualization: Sample original and predicted masks
    st.subheader('3. Images, masks and predicted masks with type2 defects')
    cnt = 0
    for index,row in X_test[(X_test.hasDefect_1 == 0) & (X_test.hasDefect_2 == 1) & (X_test.hasDefect_3 == 0) & (X_test.hasDefect_4 == 0)].iterrows() :
      if cnt < 3:
        fig, (ax1,ax2,ax3) = plt.subplots(nrows = 1,ncols = 3,figsize=(25, 6))
        img = cv2.imread(os.path.join(train_images_path,row['ImageId']))
        # pdb.set_trace()
        ax1.imshow(img)
        ax1.set_title('Original_image_'+row['ImageId'])
        
        file_name = (row['ImageId']).split('.')[0] 
        
        mask_rle_list = [row.Defect_1,row.Defect_2,row.Defect_3,row.Defect_4] 
        rgb_mask = rle_to_RGBmask(mask_rle_list, classes_tocolour, shape=(1600, 256))
        original_mask = RGBmask_to_width_height_classes(rgb_mask, colourmap)  #256 X 1600 X 5
        
        original_mask = width_height_classes_toRGB(original_mask, classes_tocolour)
        ax2.imshow(original_mask)
        ax2.set_title('Original_mask_'+file_name)
        
        img = cv2.imread(os.path.join(train_images_path,row['ImageId']))
        predict_mask = unet_model.predict(np.expand_dims(img, axis=0)).argmax(axis = -1) #256X1600X5 
        # pdb.set_trace() 
        predict_mask  = np.squeeze(predict_mask, axis=0)
        predict_mask  = one_frame_rgb(predict_mask,classes_tocolour)
        
        # predict_mask = width_height_classes_toRGB(predict_mask, classes_tocolour)
        ax3.imshow(predict_mask)
        ax3.set_title('Predicted Mask_'+os.path.split(i)[1])
        plt.show()
        st.pyplot(plt)
        cnt+=1 
      else :
        break  
    st.write('Model is very bad in detecting type 2 defects')    
    
## Visualization: Sample original and predicted masks
    st.subheader('4. Images, masks and predicted masks with type3 defects')
    cnt = 0
    for index,row in X_test[(X_test.hasDefect_1 == 0) & (X_test.hasDefect_2 == 0) & (X_test.hasDefect_3 == 1) & (X_test.hasDefect_4 == 0)].iterrows() :
      if cnt < 3:
        fig, (ax1,ax2,ax3) = plt.subplots(nrows = 1,ncols = 3,figsize=(25, 6))
        img = cv2.imread(os.path.join(train_images_path,row['ImageId']))
        # pdb.set_trace()
        ax1.imshow(img)
        ax1.set_title('Original_image_'+row['ImageId'])
        
        file_name = (row['ImageId']).split('.')[0] 
        
        mask_rle_list = [row.Defect_1,row.Defect_2,row.Defect_3,row.Defect_4] 
        rgb_mask = rle_to_RGBmask(mask_rle_list, classes_tocolour, shape=(1600, 256))
        original_mask = RGBmask_to_width_height_classes(rgb_mask, colourmap)  #256 X 1600 X 5
        
        original_mask = width_height_classes_toRGB(original_mask, classes_tocolour)
        ax2.imshow(original_mask)
        ax2.set_title('Original_mask_'+file_name)
        
        img = cv2.imread(os.path.join(train_images_path,row['ImageId']))
        predict_mask = unet_model.predict(np.expand_dims(img, axis=0)).argmax(axis = -1) #256X1600X5 
        # pdb.set_trace() 
        predict_mask  = np.squeeze(predict_mask, axis=0)
        predict_mask  = one_frame_rgb(predict_mask,classes_tocolour)
        
        # predict_mask = width_height_classes_toRGB(predict_mask, classes_tocolour)
        ax3.imshow(predict_mask)
        ax3.set_title('Predicted Mask_'+os.path.split(i)[1])
        plt.show()
        st.pyplot(plt)
        cnt+=1 
      else :
        break  
    st.write('Model is performing very well in detecting type 3 defects')     

## Visualization: Sample original and predicted masks
    st.subheader('5. Images, masks and predicted masks with type4 defects')
    cnt = 0
    for index,row in X_test[(X_test.hasDefect_1 == 0) & (X_test.hasDefect_2 == 0) & (X_test.hasDefect_3 == 0) & (X_test.hasDefect_4 == 1)].iterrows() :
      if cnt < 3:
        fig, (ax1,ax2,ax3) = plt.subplots(nrows = 1,ncols = 3,figsize=(25, 6))
        img = cv2.imread(os.path.join(train_images_path,row['ImageId']))
        # pdb.set_trace()
        ax1.imshow(img)
        ax1.set_title('Original_image_'+row['ImageId'])
        
        file_name = (row['ImageId']).split('.')[0] 
        
        mask_rle_list = [row.Defect_1,row.Defect_2,row.Defect_3,row.Defect_4] 
        rgb_mask = rle_to_RGBmask(mask_rle_list, classes_tocolour, shape=(1600, 256))
        original_mask = RGBmask_to_width_height_classes(rgb_mask, colourmap)  #256 X 1600 X 5
        
        original_mask = width_height_classes_toRGB(original_mask, classes_tocolour)
        ax2.imshow(original_mask)
        ax2.set_title('Original_mask_'+file_name)
        
        img = cv2.imread(os.path.join(train_images_path,row['ImageId']))
        predict_mask = unet_model.predict(np.expand_dims(img, axis=0)).argmax(axis = -1) #256X1600X5 
        # pdb.set_trace() 
        predict_mask  = np.squeeze(predict_mask, axis=0)
        predict_mask  = one_frame_rgb(predict_mask,classes_tocolour)
        
        # predict_mask = width_height_classes_toRGB(predict_mask, classes_tocolour)
        ax3.imshow(predict_mask)
        ax3.set_title('Predicted Mask_'+os.path.split(i)[1])
        plt.show()
        st.pyplot(plt)
        cnt+=1 
      else :
        break  
    st.write('Model performace in detecting type 4 defects is not bad') 
    
 
    st.write('**Observation:** Above plots show that, the model prediction order is defect3 > defect 1 > defect4 > defect2 and this order order is same as the distribution of defects in the given data.')
    
    st.header('Confusion matrix')
    st.subheader('1. Confusion matrix for images with no defects')         
    img = cv2.imread('cfm_nodefect.PNG',3)
    b,g,r = cv2.split(img)           # get b, g, r
    img = cv2.merge([r,g,b])     # switch it to r, g, b
    plt.figure(figsize = (15,2))
    plt.imshow(img)
    plt.show()
    st.pyplot(plt)
    
    st.subheader('2. Confusion matrix for images with type1 defects')         
    img = cv2.imread('cfm_defect1.PNG',3)
    b,g,r = cv2.split(img)           # get b, g, r
    img = cv2.merge([r,g,b])     # switch it to r, g, b
    plt.imshow(img)
    plt.show()
    st.pyplot(plt)
    
    st.subheader('3. Confusion matrix for images with type2 defects')         
    img = cv2.imread('cfm_defect2.PNG',3)
    b,g,r = cv2.split(img)           # get b, g, r
    img = cv2.merge([r,g,b])     # switch it to r, g, b
    plt.imshow(img)
    plt.show()
    st.pyplot(plt)    
    
    st.subheader('4. Confusion matrix for images with type3 defects')         
    img = cv2.imread('cfm_defect3.PNG',3)
    b,g,r = cv2.split(img)           # get b, g, r
    img = cv2.merge([r,g,b])     # switch it to r, g, b
    plt.imshow(img)
    plt.show()
    st.pyplot(plt)  
    
    st.subheader('5. Confusion matrix for images with type4 defects')         
    img = cv2.imread('cfm_defect4.PNG',3)
    b,g,r = cv2.split(img)           # get b, g, r
    img = cv2.merge([r,g,b])     # switch it to r, g, b
    plt.imshow(img)
    plt.show()
    st.pyplot(plt) 
    
    st.write('**Observation:** We can observe that except for type  2 defects every other defect type has good scores, this is because there are only few images with type2 defects')
    
    st.header('Future Work')
    st.write('1. More data with type 2 defects can be added to give enough scope for the model to train on defect 2')
    st.write('2. Individual models ca be trained train to detect the pixels for each defect type respectively')
if __name__ == "__main__":
    main()

