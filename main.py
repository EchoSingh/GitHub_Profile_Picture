import openai
import requests
from PIL import Image
from io import BytesIO
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set your OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_image(prompt, model="dall-e-3", size="1024x1024", n=1):
    """
    Generate an image using DALL-E 3 model from OpenAI based on the provided prompt.

    Parameters:
    - prompt (str): The description of the image to be generated.
    - model (str): The model to be used for image generation. Default is "dall-e-3".
    - size (str): The size of the generated image. Default is "1024x1024".
    - n (int): The number of images to generate. Default is 1.

    Returns:
    - list: A list of URLs of the generated images.
    """
    try:
        response = openai.Image.create(
            model=model,
            prompt=prompt,
            n=n,
            size=size
        )
        image_urls = [img['url'] for img in response['data']]
        logging.info(f"Generated image URLs: {image_urls}")
        return image_urls
    except Exception as e:
        logging.error(f"Failed to generate image: {e}")
        return []

def save_image(image_url, filename):
    """
    Save an image from a URL to a local file.

    Parameters:
    - image_url (str): The URL of the image to be saved.
    - filename (str): The local file name to save the image as.
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img.save(filename)
        logging.info(f"Image saved as {filename}")
    except requests.RequestException as e:
        logging.error(f"Failed to download image: {e}")
    except IOError as e:
        logging.error(f"Failed to save image: {e}")

def main():
    """
    Main function to generate and save an image based on user input.
    """
    logging.info("Starting the image generation script")

    # Get custom prompt from user
    prompt = input("Enter a description for the image to generate: ").strip()
    if not prompt:
        logging.error("Prompt cannot be empty")
        return

    # Get custom file name from user
    filename = input("Enter a file name to save the image (without extension): ").strip()
    if not filename:
        logging.error("File name cannot be empty")
        return
    filename += ".png"

    # Create the outputs_generated directory if it doesn't exist
    output_dir = "outputs_generated"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created directory: {output_dir}")

    # Full path for the output file
    full_path = os.path.join(output_dir, filename)

    # Check if file already exists
    if os.path.exists(full_path):
        logging.warning(f"File {full_path} already exists and will be overwritten")

    # Generate image
    image_urls = generate_image(prompt)
    if not image_urls:
        logging.error("No images were generated")
        return

    # Save the first image
    save_image(image_urls[0], full_path)

if __name__ == "__main__":
    main()
