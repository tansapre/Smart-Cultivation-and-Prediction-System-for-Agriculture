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
	model = load_model('AlexNetModel.hdf5')


	output_dict = {'Apple___Apple_scab': 0,
	               'Apple___Black_rot': 1,
	               'Apple___Cedar_apple_rust': 2,
	               'Apple___healthy': 3,
	               'Blueberry___healthy': 4,
	               'Cherry_(including_sour)___Powdery_mildew': 5,
	               'Cherry_(including_sour)___healthy': 6,
	               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7,
	               'Corn_(maize)___Common_rust_': 8,
	               'Corn_(maize)___Northern_Leaf_Blight': 9,
	               'Corn_(maize)___healthy': 10,
	               'Grape___Black_rot': 11,
	               'Grape___Esca_(Black_Measles)': 12,
	               'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 13,
	               'Grape___healthy': 14,
	               'Orange___Haunglongbing_(Citrus_greening)': 15,
	               'Peach___Bacterial_spot': 16,
	               'Peach___healthy': 17,
	               'Pepper,_bell___Bacterial_spot': 18,
	               'Pepper,_bell___healthy': 19,
	               'Potato___Early_blight': 20,
	               'Potato___Late_blight': 21,
	               'Potato___healthy': 22,
	               'Raspberry___healthy': 23,
	               'Soybean___healthy': 24,
	               'Squash___Powdery_mildew': 25,
	               'Strawberry___Leaf_scorch': 26,
	               'Strawberry___healthy': 27,
	               'Tomato___Bacterial_spot': 28,
	               'Tomato___Early_blight': 29,
	               'Tomato___Late_blight': 30,
	               'Tomato___Leaf_Mold': 31,
	               'Tomato___Septoria_leaf_spot': 32,
	               'Tomato___Spider_mites Two-spotted_spider_mite': 33,
	               'Tomato___Target_Spot': 34,
	               'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35,
	               'Tomato___Tomato_mosaic_virus': 36,
	               'Tomato___healthy': 37}

	output_list = list(output_dict.keys())


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
	img = image.img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = img/255
	prediction = model.predict(img)

	prediction_flatten = prediction.flatten()
	

	max_val_index = np.argmax(prediction_flatten)
	result = output_list[max_val_index]

	    
	accuracy =prediction_flatten[max_val_index]*100
	accuracy = float("{:.3f}".format(accuracy))
	    
	 

	return result, accuracy;

st.title("Plant Disease prediction")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    imageee = Image.open(uploaded_file)
    imagee = imageee.resize((224, 224))
    r,a = pred(imagee)
    st.image(imageee, caption='Uploaded Image.', use_column_width=True)
    if(st.button("Predict")):
    	st.write("")

    	if a > 80 :
	    	st.write('%s (%.2f%%)' % (r,a))

	   

 