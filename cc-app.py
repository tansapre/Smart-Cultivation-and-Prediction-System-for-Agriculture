import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import cv2
import base64
import io
import torch
import random
import operator
import SessionState
import requests
import webbrowser

from PIL import Image
from skimage import io

from google.cloud import firestore
from utils.model import ResNet9


from classify import predict
from torchvision import transforms
from flask import Flask, render_template, request, Markup

from tensorflow.keras.models import Sequential
from keras.models import load_model
global graph, model, output_list


result = True
if result:

	st.success("Logged In")
	st.write("Crop Suggestion")
	
	db1 = firestore.Client.from_service_account_json("serviceAccountKey.json")
	docs1 = list(db1.collection(u'ANAGHA').stream())
	docs_dict1 = list(map(lambda x: x.to_dict(), docs1))
	df1 = pd.DataFrame(docs_dict1)
	print(df1)
	#st.write(df1)
	n = df1['n']
	n = n.iloc[0]
	p = df1['p']
	p =p.iloc[0]
	k = df1['k']
	k =k.iloc[0]
	soil_moisture = df1['soil moisture']
	soil_moisture =soil_moisture.iloc[0]
	soil_temperature = df1['soil temperature']
	soil_temperature =soil_temperature.iloc[0]
	surrounding_temperature = df1['surrounding temperature']
	surrounding_temperature =surrounding_temperature.iloc[0]
	surrounding_humidity = df1['surrounding humidity']
	surrounding_humidity =surrounding_humidity.iloc[0]
	rainfall = df1['rainfall']
	rainfall = rainfall.iloc[0]
	ph = df1['pH']
	ph = ph.iloc[0]
	asach = pd.DataFrame(np.array([n,p,k,soil_moisture,soil_temperature,surrounding_temperature,surrounding_humidity]))
	#st.write(asach) 
	st.subheader("Crop Suggestion")
	crop_recommendation_model_path = 'models/RandomForest.pkl'
	crop_recommendation_model = pickle.load(
	open(crop_recommendation_model_path, 'rb'))


	N = st.number_input('Nitrogen',min_value=0, max_value=1000, value=n, step=1)
	P = st.number_input('Phosphorus',min_value=0, max_value=1000, value=p, step=1)
	K = st.number_input('Potassium',min_value=0, max_value=1000, value=k, step=1)
	ph = st.number_input('pH',min_value=0, max_value=14, value=ph, step=1)
	rainfall = st.number_input('Rainfall(mm)',min_value=0, max_value=1000, value=rainfall, step=1)
	soil_m = st.number_input('Soil Moisture',min_value=0, max_value=1000, value=soil_moisture, step=1)
	soil_t = st.number_input('Soil Temperature',min_value=0, max_value=1000, value=soil_temperature, step=1)
	#temperature = 
	#humidity = 

	def preds(N, P, K, ph, rainfall):

		api_key = '9d7cde1f6d07ec55650544be1631307e'
		base_url = "http://api.openweathermap.org/data/2.5/weather?"
		complete_url = base_url + "appid=" + api_key + "&q=" + 'pune'
		response = requests.get(complete_url)
		x = response.json()
		y = x['main']
		temperature = round((y["temp"] - 273.15), 2)
		humidity = y["humidity"]
		data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
		my_prediction = crop_recommendation_model.predict(data)
		final_prediction = my_prediction[0]

		return final_prediction,temperature,humidity






	#temperature = round((x["temp"] - 273.15), 2)
	#humidity = x["humidity"]
	session_state = SessionState.get(name = "", button_start = False)
	button_start = st.button('Predict')
	if button_start:
		session_state.button_start = True
		st.success("Success")
		final_prediction,temperature,humidity = preds(N, P, K, ph, rainfall)
		st.write("The city of Pune" )
		st.write('Temperature : {}C'.format(temperature))
		st.write('humidity : {}'.format(humidity)) 
		st.subheader("The Suggested crop is :")
		st.write(final_prediction)

else:
	st.write('')
