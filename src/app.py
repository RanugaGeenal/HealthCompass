import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("E:/HealthCompass/notebooks/decision_tree_model.pkl")

def get_user_input():
    st.sidebar.header('User Input Parameters')
    
    # Inputs
    age = st.number_input('Age', min_value=1, max_value=120)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, format="%.1f")
    children = st.number_input('Number of Children', min_value=0, max_value=10)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    # Convert inputs to same format as training
    sex = 1 if sex == "male" else 0
    smoker = 1 if smoker == "yes" else 0

    # Create dataframe with correct columns
    user_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'northeast': [1 if region == "northeast" else 0],
        'northwest': [1 if region == "northwest" else 0],
        'southeast': [1 if region == "southeast" else 0],
        'southwest': [1 if region == "southwest" else 0],
    })

    return user_data

def main():
    st.title("Medical Insurance Cost Prediction")

    user_input = get_user_input()

    st.subheader("User Input")
    st.write(user_input)

    # Predict
    prediction = model.predict(user_input)
    st.subheader("Predicted Medical Charges")
    st.write(f"${prediction[0]:,.2f}")

if __name__ == "__main__":
    main()
