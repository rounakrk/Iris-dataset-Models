import numpy as np
import streamlit as st
import pickle
from sklearn import tree
import pandas as pd
import PIL
import requests
from PIL import Image

st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

st.sidebar.title("Iris Parameters")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

target = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

st.subheader('User Input parameters')
st.write(df)

st.subheader('Iris categories')
st.write(target)

loaded_model = pickle.load(open('iris.sav', 'rb'))

prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.write("Prediction is :",target[prediction[0]])

if prediction[0]==0:
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Iris_setosa_2.jpg/1200px-Iris_setosa_2.jpg"
elif prediction[0]==1:
    url = "https://upload.wikimedia.org/wikipedia/commons/2/27/Blue_Flag%2C_Ottawa.jpg"
else:
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Iris_virginica_2.jpg/1200px-Iris_virginica_2.jpg"

im = Image.open(requests.get(url, stream=True).raw)
im = im.resize((500,500))
st.image(im, width = 500, )
