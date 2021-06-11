import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from google.cloud import firestore

db = firestore.Client.from_service_account_json("serviceAccountKey.json")

docs = list(db.collection(u'APOORVA').stream())

docs_dict = list(map(lambda x: x.to_dict(), docs))
df = pd.DataFrame(docs_dict)
df.to_csv('out.csv')
user = df['username']
user =user.iloc[0]
pswd = df['password']
print(user)