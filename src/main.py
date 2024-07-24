import streamlit as st
import base64
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
engine_id = "stable-diffusion-v1-6"
api_host = os.getenv('API_HOST', 'https://api.stability.ai')
api_key = os.getenv("STABILITY_API_KEY")

# Check if API key is set
if api_key is None:
    st.error("Missing Stability API key.")
    st.stop()

# Ensure output directory exists
output_dir = "./"
os.makedirs(output_dir, exist_ok=True)

# Load custom CSS
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Streamlit app
st.markdown("<div class='header'>✨ Generate Your Custom GitHub Profile Picture! ✨</div>", unsafe_allow_html=True)
st.markdown("<div class='content'>Create an anime-style GitHub profile picture that reflects your personality and passion for coding. 🚀👨‍💻</div>", unsafe_allow_html=True)

# User input for the text prompt
user_prompt = st.text_area("Describe your GitHub profile picture:", 
    "Create an anime-style GitHub profile picture for a boy. The character should have a friendly and approachable expression, embodying a sense of curiosity and enthusiasm, reflecting the qualities of a passionate coder. Incorporate elements that suggest technology or coding, such as a pair of stylish glasses, a laptop, or a background with subtle code or tech motifs. Use vibrant and appealing colors to make the profile picture stand out and convey a sense of creativity and innovation.")

if st.button("Generate Image"):
    with st.spinner('Generating image...'):
        payload = {
            "text_prompts": [{"text": user_prompt}],
            "cfg_scale": 7,
            "height": 1024,
            "width": 1024,
            "samples": 1,
            "steps": 30,
        }

        # Send the request to Stability AI
        response = requests.post(
            f"{api_host}/v1/generation/{engine_id}/text-to-image",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json=payload,
        )

        # Check for a successful response
        if response.status_code != 200:
            st.error(f"Non-200 response: {response.text}")
            st.stop()

        # Process and display the generated images
        data = response.json()
        for i, image in enumerate(data["artifacts"]):
            image_data = base64.b64decode(image["base64"])
            file_path = os.path.join(output_dir, f"aditya_profile_pic_sdx_1{i}.png")
            with open(file_path, "wb") as f:
                f.write(image_data)
            st.image(file_path, caption=f"Generated Image {i+1}")
            st.success(f"Saved image to {file_path}")

    st.balloons()
