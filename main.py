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
    raise Exception("Missing Stability API key.")

# Ensure output directory exists
output_dir = "./outputs_generated"
os.makedirs(output_dir, exist_ok=True)

# Request payload
payload = {
    "text_prompts": [
        {"text": "Create an anime-style GitHub profile picture for a boy. The character should have a friendly and approachable expression, embodying a sense of curiosity and enthusiasm, reflecting the qualities of a passionate coder. Incorporate elements that suggest technology or coding, such as a pair of stylish glasses, a laptop, or a background with subtle code or tech motifs. Use vibrant and appealing colors to make the profile picture stand out and convey a sense of creativity and innovation."}
    ],
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
    raise Exception("Non-200 response: " + str(response.text))

# Process and save the generated images
data = response.json()
for i, image in enumerate(data["artifacts"]):
    image_data = base64.b64decode(image["base64"])
    file_path = os.path.join(output_dir, f"aditya_profile_pic_sdx_1{i}.png")
    with open(file_path, "wb") as f:
        f.write(image_data)
    print(f"Saved image to {file_path}")
