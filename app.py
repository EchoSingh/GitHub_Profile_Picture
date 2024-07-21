import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import os

# Load Stable Diffusion model
@st.cache_resource
def load_model():
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    pipe.to(device)
    return pipe

# Load the model
pipe = load_model()

# Ensure output directory exists
output_dir = "./outputs_generated"
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
        images = pipe([user_prompt], num_inference_steps=50)["images"]

        for i, image in enumerate(images):
            file_path = os.path.join(output_dir, f"aditya_profile_pic_{i}.png")
            image.save(file_path)
            st.image(file_path, caption=f"Generated Image {i+1}")
            st.success(f"Saved image to {file_path}")

    st.balloons()
