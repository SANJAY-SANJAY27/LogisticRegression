import streamlit as st
import pandas as pd
import pickle

# Page config
st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")

# Load model
@st.cache_resource
def load_model():
    with open("logistic_titanic_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Title
st.title("🚢 Titanic Survival Prediction")
st.write("Predict whether a passenger would survive")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
fare = st.number_input("Fare", min_value=0.0, value=32.0)
embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

# Encoding (MUST match training)
sex_encoded = 1 if sex == "male" else 0
embarked_map = {"C": 0, "Q": 1, "S": 2}
embarked_encoded = embarked_map[embarked]

# Input dataframe
input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex_encoded],
    "Age": [age],
    "Fare": [fare],
    "Embarked": [embarked_encoded]
})

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("🎉 Passenger is likely to SURVIVE")
    else:
        st.error("❌ Passenger is NOT likely to survive")


