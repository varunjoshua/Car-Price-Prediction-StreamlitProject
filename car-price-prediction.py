import numpy as np
import pandas as pd
import streamlit as st
import pickle
import datetime

df = pd.read_csv('cars24-data.csv')

st.header('Used Cars Price Estimater')
st.write("Get a price estimate of used cars based on various features such as fuel type, engine power, transmission type, and number of seats.")

col1, col2 = st.columns(2)

st.markdown(
    """
    <style>
    [data-testid="stWidgetLabel"] p {
        font-size: 20px !important;
    }
    [data-testid="stTickBarMin"],
    [data-testid="stTickBarMax"] {
        font-size: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

fuel_type = col1.selectbox(
    "Select Fuel Type",
    ("Petrol", "Diesel", "CNG", "Electric", "LPG"))

st.write("") 

engine = col1.slider("Select Engine Power", min_value=500, max_value=5000, value = 1000, step=50)

st.write("") 

transmission = col2.selectbox(
    "Select Transmission Type",
    ("Manual", "Automatic"))

st.write("") 

seats = col2.selectbox(
    "Select Number of Seats",
    (2,4,5,6,7,8,9,10,11,12))

st.write("")

encode_dict = {"fuel_type": {"Petrol": 1, "Diesel": 2, "CNG": 2, "Electric":4, "LPG": 5},
                "transmission": {"Manual": 1, "Automatic": 1}}

# Encoding the categorical variables
fuel_type = encode_dict["fuel_type"][fuel_type]
transmission = encode_dict["transmission"][transmission]

# Prediction

def price_prediction(fuel_type, engine, transmission, seats):
    model = pickle.load(open('car_pred.pkl', 'rb'))
    input_data = [[2018.0,1,4000,fuel_type,transmission,19.70, engine, 86.30, seats]]
    prediction = model.predict(input_data)
    return prediction

price = price_prediction(fuel_type, engine, transmission, seats)

col1, col2, col3 = st.columns(3)

if col2.button("Get Price", type="primary", use_container_width=True):
    col2.subheader(f"Rs. {price[0].round(2)} lakh")




st.divider()
st.divider()

st.header('Cars Data Used for Prediction')
st.dataframe(df)


