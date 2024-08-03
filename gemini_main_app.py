from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
import google.generativeai as genai

load_dotenv() # loads all the env variables from .env
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Function to load Gemini pro vision
# model = genai.GenerativeModel('gemini-pro-vision')
model = genai.GenerativeModel('gemini-1.5-pro-latest')

def get_gemini_response(input, image, prompt):
    response = model.generate_content([input, image[0], prompt])
    return response.text


def input_image_details(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")
    

st.set_page_config(page_title="Brain Tumor MRI Image Extractor Chatbot")
st.header("Brain Tumor MRI Image Extractor Chatbot")
input = st.text_input("Input Prompt:", key="input")
submit = st.button("Submit")

uploaded_file = st.file_uploader("Choose as image...", type=["jpg", "jpeg", "png"])

image = " "
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

input_prompt = """
You are an expert of understanding medical MRI images. We will upload a images of Brain Tumor and you will have to answer any questions based on the uploaded image.
Please note:
- You are specifically focused on brain tumor MRI images. For accurate analysis and assistance, ensure that we upload only MRI images related to brain tumors.
- You are here to provide insights and answer questions specifically related to brain tumors identified in MRI images. Unfortunately, You cannot assist with queries not related to brain tumors. 
- If you find MRI Images which is not related to Brain tumor, Say that "Sorry!!! Im unable to answer your queries which is not related to Brain Tumor. Please upload only Brain Tumor MRI Images, Im a Brain Tumor MRI Image Extractor Chatbot!"
"""

# After clicking submit button
if submit:
    image_data = input_image_details(uploaded_file)
    response = get_gemini_response(input_prompt, image_data, input)
    st.subheader("The Response is:")
    st.write(response)