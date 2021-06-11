"""
# -*- coding: utf-8 -*-


Created on Sat May  8 13:40:01 2021

@author: 91758

"""
import streamlit as st
import numpy as np
import pandas as pd
import requests
import webbrowser

from google.cloud import firestore



st.title("Smart Cultivation and Prediction System for Agriculture")
menu = ["Login","SignUp"]
choice = st.selectbox("Menu",menu)

if choice == "Login":
        username= st.text_input("User Name")
        password= st.text_input("Password", type='password')
        if st.button("Login"):
                if not (username):
                        st.warning("Enter Valid Username")
                elif not (password):
                        st.warning("Enter Valid Password")
                else:
                    url = 'http://localhost:8502'
                    webbrowser.open_new_tab(url)
elif choice == "SignUp":
        new_user = st.text_input("Username")
        new_password = st.text_input("Password",type='password')
        confirmpassword = st.text_input("Confirm Password",type='password')

        if st.button("Signup"):
            if not (new_user):
                st.warning("Enter Valid Username")
            elif not (new_password):
                st.warning("Enter Valid Password")
            elif (new_password!= confirmpassword):
                st.warning("Password does not match")
            else:
                db = firestore.Client.from_service_account_json("serviceAccountKey.json")
                data={"username":new_user,"password":new_password}
                doc_name=new_user+"_final_record"
                db.collection(new_user).document(doc_name).set(data)
                st.success("You have successfully created a valid Account")
                st.info("Go to Login Menu to login")
            
