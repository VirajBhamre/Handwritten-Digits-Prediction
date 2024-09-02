import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

model = load_model("Number_Predictor.keras")

st.title("Handwritten Digit Prediction")

uploaded_file = st.file_uploader("Upload a handwritten digit image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    img = image.resize((28, 28))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 784)
    
    predictions = model.predict(img_array)
    predicted_digit = np.argmax(predictions)
    
    st.write("Predicted Digit:", predicted_digit)
