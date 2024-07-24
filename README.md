# 📸 GitHub Profile Picture Generator

![Generated Image](outputs_generated/adi.png)

✨ Generate Your Custom GitHub Profile Picture! ✨

## 📝 Overview

This project allows you to create an anime-style GitHub profile picture that reflects your personality and passion for coding. The image generation is done using a diffusion model pipeline and a text generation model to extend the prompts.

## 🌟 Features

- 🎨 Generate detailed anime-style profile pictures.
- 🔧 Customize the generation process with various parameters.
- 💾 Save the generated images in the `outputs_generated` folder.

## ⚙️ Installation

### Prerequisites 📋

Make sure you have the following software installed:

- [Python](https://www.python.org/)
- [Git](https://git-scm.com/)

1. Clone the repository:

    ```bash
    git clone https://github.com/EchoSingh/GitHub_Profile_Picture.git 
    cd ChatBot_WebProject
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

1. Run the Streamlit app:

    ```bash
    streamlit run src/app.py
    ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Describe your GitHub profile picture and adjust the parameters as needed.

4. Click the "Generate Image" button to create your custom profile picture.

5. The generated image will be displayed and saved in the `outputs_generated` folder.

## 📦 Dependencies

- `streamlit`
- `torch`
- `numpy`
- `diffusers`
- `transformers`
- `Pillow`


## Use the Link and generate images ..
- Lowest Quality Images : (src/main_SDX.py)
- Its best if u use API as it give initial 25 credits (src/main.py).
- Using CPU for generating so your image may not as good as expected using link (src/app.py)
- [RUNNING LINK](https://huggingface.co/spaces/adi2606/Profile_Pic_Generator) (using now model : stabilityai/sdxl-turbo )

## Files and Directories 📁
```
📁 project/
├── 📂 src/
│   ├── 📄 app.py
│   ├── 📄 main.py
│   └── 📄 main_SDX.py
├── 📂 outputs_generated/
│   ├── 🖼️ adi.png
│   ├── 🖼️ more image (4 days ago)
│   ├── 🖼️ aditya_profile_pic.png (first commit, last week)
│   └── 🖼️ aditya_profile_pic_sdx_10.png (sdx_api, last week)
├── 📄 .gitignore
├── 📄 LICENSE
├── 📄 README.md
└── 📄 requirements.txt
```

## Contributing 🤝

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. **Fork the Project**:
   Click on the `Fork` button at the top right corner of the repository page.

2. **Create a Branch**:
   ```bash
   git checkout -b feature/AmazingFeature
   ```

3. **Commit your Changes**:
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```

4. **Push to the Branch**:
   ```bash
   git push origin feature/AmazingFeature
   ```

5. **Open a Pull Request**:
   Click on the `New Pull Request` button and provide details of your changes.

## License 📄

Distributed under the MIT License. See `LICENSE` for more information.

## Contact 📬

- **Name**: Aditya
- **GitHub**: [EchoSingh](https://github.com/EchoSingh)

## Acknowledgements 🙌

- Thanks to all the contributors who have helped in improving this project.
- Special thanks to the open source community for their continuous support and inspiration.

## Support ⭐
- If you like this project, please give it a star on GitHub! Your support is appreciated and helps to keep the project growing.
---
Made with ❤️ by Aditya
