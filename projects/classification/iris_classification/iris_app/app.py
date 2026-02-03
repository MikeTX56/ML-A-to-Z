import streamlit as st
import joblib
import numpy as np
import os

# Get the directory that this script (app.py) is in
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'iris_random_forest_model.joblib')
@st.cache_resource
def load_model():
    return joblib.load(model_path)
model = load_model()
st.title("Iris Species Predictor")
# 1. Manually set ranges based on our original data we used to train
sepal_length = st.slider("sepal length (cm)", 4.3, 7.9, 5.8)
sepal_width = st.slider("sepal width (cm)", 2.0, 4.4, 3.0)
petal_length = st.slider("petallength  (cm)",1.0, 6.9, 3.7)
petal_width = st.slider("petal width (cm)", 0.1, 2.5, 1.2)

features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button('Predict Species'):
    prediction = model.predict(features)

    species = ['Setosa', 'Versicolor', 'Virginica']
    result = species[prediction[0]]
    
    st.success(f"The predicted species is: **{result}**")
