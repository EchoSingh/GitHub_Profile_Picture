import streamlit as st
import torch
import numpy as np
from diffusers import DiffusionPipeline
from transformers import pipeline

# Load text generation pipeline
text_pipe = pipeline('text-generation', model='daspartho/prompt-extend')

def extend_prompt(prompt):
    return text_pipe(prompt + ',', num_return_sequences=1, max_new_tokens=50)[0]["generated_text"]

@st.cache_resource
def load_pipeline(use_cuda):
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True
    )
    if device == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
    pipe.to(device)
    return pipe

def generate_image(prompt, use_details, steps, seed, use_cuda):
    pipe = load_pipeline(use_cuda)
    generator = torch.manual_seed(seed) if seed != 0 else torch.manual_seed(np.random.randint(0, 2**32))
    extended_prompt = extend_prompt(prompt) if use_details else prompt
    image = pipe(prompt=extended_prompt, generator=generator, num_inference_steps=steps, guidance_scale=7.5).images[0]
    return image, extended_prompt

# Custom CSS styling
st.markdown("""
    <style>
        body {background-color: #E8F5E9;}
        .header {font-size: 2.5em; color: #2E7D32; text-align: center; margin-top: 20px;}
        .subheader {font-size: 1.2em; color: #4CAF50; text-align: center; margin-bottom: 30px;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'>✨ Generate Your Custom GitHub Profile Picture! ✨</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Create an anime-style GitHub profile picture that reflects your personality and passion for coding. 🚀👨‍💻</div>", unsafe_allow_html=True)

# Input widgets
input_text = st.text_area("Describe your GitHub profile picture:",
    "Create an anime-style GitHub profile picture for a boy. The character should have a friendly and approachable expression, embodying a sense of curiosity and enthusiasm, reflecting the qualities of a passionate coder. Incorporate elements that suggest technology or coding, such as a pair of stylish glasses, a laptop, or a background with subtle code or tech motifs. Use vibrant and appealing colors to make the profile picture stand out and convey a sense of creativity and innovation.")

details_checkbox = st.checkbox("Generate Details?", value=True)
steps_slider = st.slider("Number of Iterations", min_value=1, max_value=50, value=25, step=1)
seed_slider = st.slider("Seed", min_value=0, max_value=9007199254740991, value=3982317, step=1) 
cuda_checkbox = st.checkbox("Use CUDA?", value=False)

if st.button("Generate Image"):
    with st.spinner('Generating image...'):
        image, extended_prompt = generate_image(input_text, details_checkbox, steps_slider, seed_slider, cuda_checkbox)
        st.image(image, caption="Generated Image")
        st.text(f"Extended Prompt: {extended_prompt}")
    st.balloons()
