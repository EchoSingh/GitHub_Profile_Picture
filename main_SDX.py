
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import torch
from diffusers import StableDiffusionPipeline



def get_sd_pipeline():
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.to(device)
    return pipe



def get_ai_generated_pattern(pipe, prompt="an anime github profile picture "):
    result = pipe(prompt, guidance_scale=7.5)
    image = result.images[0]
    return image



pipe = get_sd_pipeline()

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ai_pattern = get_ai_generated_pattern(pipe)
ax.imshow(ai_pattern, extent=[0, 1, 0, 1], aspect='auto')

ax.axis('off')

output_path = 'outputs_generated/math_sacred_geometry_profile_pic_v2.png'
fig.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)

# Display the image
plt.show()

print(f"Profile picture generated and saved to {output_path}")