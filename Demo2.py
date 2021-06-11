import streamlit as st 
from PIL import Image
from classify import predict

import numpy as np
import pickle
import cv2
import base64
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from skimage import io
from keras.preprocessing import image
 
import tensorflow as tf
from keras.models import load_model
global graph, model, output_list


def pred(image1):
	model = load_model('model.h5')
	disease_class = ['Pepper__bell___Bacterial_spot',
					'Pepper__bell___healthy',
					'Potato___Early_blight',
					'Potato___Late_blight',
					'Potato___healthy',
					'Tomato_Bacterial_spot',
					'Tomato_Early_blight',
					'Tomato_Late_blight',
					'Tomato_Leaf_Mold',
					'Tomato_Septoria_leaf_spot',
					'Tomato_Spider_mites_Two_spotted_spider_mite',
					'Tomato__Target_Spot',
					'Tomato__Tomato_YellowLeaf__Curl_Virus',
					'Tomato__Tomato_mosaic_virus',
					'Tomato_healthy']


	default_image_size = tuple((256, 256))
	def convert_image_to_array(image_dir):
	    try:
	        image = cv2.imread(image_dir)
	        if image is not None :
	            image = cv2.resize(image, default_image_size)   
	            return img_to_array(image)
	        else :
	            return np.array([])
	    except Exception as e:
	        print(f"Error : {e}")
	        return None


	img = image1
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis = 0)
	#x = np.array(x, 'float32')
	x /= 255
	prediction = model.predict(x)
	custom = model.predict(x)
	#x = x.reshape([64, 64]);
	#plt.gray()
	

	a=custom[0]
	ind=np.argmax(a)
	acc = a[ind]*100     
	pred = disease_class[ind]

	return pred,acc;  	    
	

st.title("Plant Disease prediction")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    imageee = Image.open(uploaded_file)
    imagee = imageee.resize((64, 64))
    st.image(imageee, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Predicting...")
    r,a = pred(imagee)
    if a > 80:
    	st.write('%s (%.2f%%)' % (r,a))
    else :
   		st.write('Could not recognise image')
   		st.write('The model predicts:')
   		st.write('%s (%.2f%%)' % (r,a))		

 
