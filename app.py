import streamlit as st
from nutrition_agent import nutrition_agent
from PIL import Image
import requests
from io import BytesIO
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config(page_title="Smartest AI Nutrition Assistant", layout="wide")

st.title("🥗 Smartest AI Nutrition Assistant")

# Tabs
tab1, tab2 = st.tabs(["📋 Get Personalized Diet Plan", "📷 Image-Based Nutrition Analysis"])

# ---------------- Tab 1: Personalized Diet Plan ----------------
with tab1:
    st.subheader("📋 Fill the Form to Get Your Personalized Diet Plan")

    with st.form("diet_form"):
        name = st.text_input("What is your name?")
        age = st.text_input("Age")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        diet_type = st.selectbox("Are you Vegetarian or Non-Vegetarian?", ["Vegetarian", "Non-Vegetarian"])
        location = st.text_input("Geographical Location (e.g., India, US)")
        health_issues = st.text_input("Any existing health complaints?")
        health_goals = st.text_input("Your health/fitness goals?")
        likes = st.text_input("Foods you like")
        dislikes = st.text_input("Foods you dislike")
        calories = st.text_input("Approximate daily calorie requirement (if known)")

        submitted = st.form_submit_button("Generate Diet Plan")

    if submitted:
        user_profile = (
            f"My name is {name}. I am {age} years old, {gender}, {diet_type}, living in {location}. "
            f"I have the following health issues: {health_issues}. "
            f"My goal is to {health_goals}. I like eating {likes} but dislike {dislikes}. "
            f"My daily calorie need is {calories}. "
            "Please generate a personalized meal plan broken into breakfast, lunch, snacks, dinner. "
            "Also include a clear explanation of nutritional content and a conclusion for health advice."
        )

        with st.spinner("Generating your personalized meal plan..."):
            result = nutrition_agent.run(user_profile)

    # Clean output
            if result:
                if isinstance(result, str):
                    st.markdown(result)
                elif hasattr(result, 'content'):
                    st.markdown(result.content)
                else:
                    st.write(result)


# ---------------- Tab 2: Image-Based Nutrition Analysis ----------------
with tab2:
    st.subheader("📷 Upload or Paste a Food Image(JPG/JPEG/PNG only) URL")

    image = None  # Initialize image variable
    image_input_type = st.radio("Choose input method:", ["Upload Image", "Paste Image URL"])

    if image_input_type == "Upload Image":
        uploaded_file = st.file_uploader("Upload a food image", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
            except Exception as e:
                st.error(f"Error opening image: {e}")
    else:
        image_url = st.text_input("Enter image URL")
        if image_url:
            try:
                response = requests.get(image_url)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            except Exception as e:
                st.error(f"Error loading image: {e}")

    if image is not None:
        st.image(image, caption="Selected Image", use_column_width=True)

        if st.button("Analyze Nutrition"):
            prompt = (
                "Analyze the nutritional content of the food shown in this image. "
                "Identify the type of food, estimate calories, and mention if it is healthy or not."
            )
            try:
                model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
                with st.spinner("Analyzing image..."):
                    result = model.generate_content([prompt, image])
                st.markdown("### 🧠 Nutrition Analysis Result")
                st.markdown(result.text)
            except Exception as e:
                st.error(f"Failed to analyze image: {e}")
    else:
        st.info("Please upload an image or paste a valid image URL to analyze.")
