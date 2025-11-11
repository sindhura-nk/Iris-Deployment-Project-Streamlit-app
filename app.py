# import all the packages
from warnings import filterwarnings
filterwarnings('ignore')
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Build the interface
st.set_page_config(page_title='Iris Project',layout='wide')

# Add title to the page
st.title('Iris Project - Sindhura')

# Add options for the user to provide inputs
sep_len = st.number_input('Sepal Length: ',min_value=0.00,step=0.01)
sep_wid = st.number_input('Sepal Width: ',min_value=0.00,step=0.01)
pet_len = st.number_input('Petal Length: ',min_value=0.00,step=0.01)
pet_wid = st.number_input('Petal Width: ',min_value=0.00,step=0.01)

# Add a button- Action of the button is to perform prediction
submit = st.button('Predict')

# Load the pickle files: preprocessor and model files
with open("pre.pkl","rb") as file1:
    pre = pickle.load(file1)

with open("model.pkl","rb") as file2:
    model = pickle.load(file2)

# Logic for prediction
if submit:
    # Convert the data into dataframe
    dct = {
        'sepal_length':[sep_len],
        'sepal_width':[sep_wid],
        'petal_length':[pet_len],
        'petal_width':[pet_wid]
    }
    # Convert above dictionary to dataframe
    xnew = pd.DataFrame(dct)
    # Preprocess the data
    xnew_pre = pre.transform(xnew)
    # Predict the results
    preds = model.predict(xnew_pre)
    probs = model.predict_proba(xnew_pre)
    max_prob = np.max(probs)
    # Print above results
    st.subheader('Predictions are : ')
    st.subheader(f'Predictions Species: {preds[0]}')
    st.subheader(f'Probability: {max_prob*100:.2f}%')
    st.progress(max_prob)