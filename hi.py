import streamlit as st
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

# Title
st.title("🧠 Diabetes Progression Prediction")
st.write("This app uses a Random Forest model to predict diabetes disease progression based on medical data.")

# Load the dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Sidebar inputs
st.sidebar.header("Input Patient Features")
def user_input_features():
    data = {}
    for feature in diabetes.feature_names:
        data[feature] = st.sidebar.slider(
            label=feature,
            min_value=float(X[feature].min()),
            max_value=float(X[feature].max()),
            value=float(X[feature].mean())
        )
    return pd.DataFrame([data])

input_df = user_input_features()

# Display user inputs
st.subheader("User Input Features")
st.write(input_df)

# Make prediction
prediction = model.predict(input_df)

# Show prediction
st.subheader("Predicted Disease Progression")
st.write(f"📊 Prediction Score: {prediction[0]:.2f}")
