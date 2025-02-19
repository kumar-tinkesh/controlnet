# from flask_ngrok import run_with_ngrok
# from flask import Flask, render_template, request
# import torch
# import base64
# from io import BytesIO
# from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
# from pyngrok import ngrok

# # Check if CUDA is available
# device = "cuda" if torch.cuda.is_available() else "cpu"
# dtype = torch.float16 if device == "cuda" else torch.float32  # Adjust dtype

# # Load ControlNet model
# controlnet = ControlNetModel.from_pretrained(
#     "lllyasviel/sd-controlnet-canny", 
#     torch_dtype=dtype  # Use adjusted dtype
# )

# # Load Stable Diffusion with ControlNet
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", 
#     controlnet=controlnet, 
#     torch_dtype=dtype  # Use adjusted dtype
# )

# pipe.to(device)
# port_no = 5000

# # Start Flask app
# app = Flask(__name__)
# ngrok.set_auth_token("2jK44YcMq9wfqWna2tpf4gkZxCY_24fLpcRmNfyZqQQhcyJ5M")
# public_url =  ngrok.connect(port_no).public_url

# @app.route('/')
# def initial():
#     return render_template('index.html')

# print(f"To acces the Gloable link please click {public_url}")

# @app.route('/submit-prompt', methods=['POST'])
# def generate_image():
#     try:
#         prompt = request.form['prompt-input']
#         print(f"Generating an image of: {prompt}")

#         # Generate an image
#         image = pipe(prompt).images[0]
#         print("Image generated! Converting to base64...")

#         # Convert image to base64
#         buffered = BytesIO()
#         image.save(buffered, format="PNG")
#         img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
#         img_str = f"data:image/png;base64,{img_str}"

#         print("Sending image...")
#         return render_template('index.html', generated_image=img_str)

#     except Exception as e:
#         print(f"Error: {e}")
#         return render_template('index.html', error="Failed to generate image. Check server logs.")

# if __name__ == '__main__':
#     app.run(port=port_no)


# pose extraction ------------------------------------------------------------------------------------------------


# from flask_ngrok import run_with_ngrok
# from flask import Flask, render_template, request
# import torch
# import base64
# from io import BytesIO
# from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
# from controlnet_aux import OpenposeDetector
# from diffusers import UniPCMultistepScheduler
# from PIL import Image
# from IPython.display import display
# from pyngrok import ngrok
# import requests
# from io import BytesIO

# # Check if CUDA is available
# device = "cuda" if torch.cuda.is_available() else "cpu"
# dtype = torch.float16 if device == "cuda" else torch.float32  # Adjust dtype

# # Initialize Flask app
# app = Flask(__name__)

# # ngrok setup for public URL
# ngrok.set_auth_token("2jK44YcMq9wfqWna2tpf4gkZxCY_24fLpcRmNfyZqQQhcyJ5M")
# public_url = ngrok.connect(5000).public_url

# print(f"To access the global link, please click {public_url}")

# def generate_images(generate_prompt, image_url, model_id="runwayml/stable-diffusion-v1-5"):
#     # Download the image from the provided URL
#     response = requests.get(image_url)
#     input_image = Image.open(BytesIO(response.content))

#     # Load Openpose model for property extraction
#     pose_model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
#     pose_image = pose_model(input_image)

#     # Load ControlNet model
#     controlnet = ControlNetModel.from_pretrained(
#         "fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float16
#     )

#     # Load Stable Diffusion pipeline with ControlNet
#     pipe = StableDiffusionControlNetPipeline.from_pretrained(
#         model_id,
#         controlnet=controlnet,
#         torch_dtype=torch.float16,
#     )
#     pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
#     pipe.enable_model_cpu_offload()
#     pipe.enable_xformers_memory_efficient_attention()

#     # Generate new images
#     generator = torch.Generator(device="cpu").manual_seed(2)
#     output = pipe(
#         [generate_prompt] * 4,  # Generate multiple images
#         [pose_image] * 4,
#         generator=[torch.Generator(device="cpu").manual_seed(i) for i in range(4)],
#         num_inference_steps=20,
#     )

#     images = output.images  # Store all images in a variable
#     return images  # Return all generated images

# @app.route('/')
# def initial():
#     return render_template('index.html')

# @app.route('/submit-prompt', methods=['POST'])
# def generate_image():
#     try:
#         # Get the prompt submitted from the form
#         prompt = request.form['prompt-input']
#         print(f"Generating an image for: {prompt}")

#         # Image URL to be used for pose extraction
#         image_url = "https://images.template.net/wp-content/uploads/2016/11/29093829/Tree-and-the-Moon-Drawing.jpg?width=530"

#         # Generate images using the provided prompt and image URL
#         images = generate_images(prompt, image_url)
#         print("Images generated! Converting to base64...")

#         # Convert the first generated image to base64
#         buffered = BytesIO()
#         images[0].save(buffered, format="PNG")
#         img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
#         img_str = f"data:image/png;base64,{img_str}"

#         print("Sending image...")
#         return render_template('index.html', generated_image=img_str)

#     except Exception as e:
#         print(f"Error: {e}")
#         return render_template('index.html', error="Failed to generate image. Check server logs.")

# if __name__ == '__main__':
#     app.run(port=5000)


# text-to-image generation ------------------------------------------------------------------------------------------------

from flask import Flask, render_template, request
import torch
import base64
import requests
from io import BytesIO
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
from pyngrok import ngrok

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32  # Adjust dtype

# Initialize Flask app
app = Flask(__name__)

# ngrok setup for public URL
ngrok.set_auth_token("2jK44YcMq9wfqWna2tpf4gkZxCY_24fLpcRmNfyZqQQhcyJ5M")  # Replace with your actual token
public_url = ngrok.connect(5000).public_url
print(f"Access the global link: {public_url}")

# Load the ControlNet model
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=dtype
).to(device)

# Load the Stable Diffusion pipeline with ControlNet
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=dtype
).to(device)


def load_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


def generate_image_from_reference(prompt, image_url):
    input_image = load_image(image_url)  # Load reference image

    output = pipeline(
        prompt=prompt,
        image=input_image,
        negative_prompt="blurry, low quality, distorted",
    )

    generated_image = output.images[0]
    return generated_image


@app.route('/')
def initial():
    return render_template('index.html')


@app.route('/submit-prompt', methods=['POST'])
def generate_image():
    try:
        prompt = request.form['prompt-input']
        print(f"Generating an image for: {prompt}")

        image_url = "https://cdn-lfs.hf.co/repos/78/0e/780e181e22f54ea1cfedf27b25482cdbb06bf28707d402dd7dd3687e4ec3e8b3/5dcb199dbfb5e1b046996d07798d26e1555cb3e8816299384377a9bc377af4f6?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27starbucks_logo.jpeg%3B+filename%3D%22starbucks_logo.jpeg%22%3B&response-content-type=image%2Fjpeg&Expires=1739955943&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczOTk1NTk0M319LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5oZi5jby9yZXBvcy83OC8wZS83ODBlMTgxZTIyZjU0ZWExY2ZlZGYyN2IyNTQ4MmNkYmIwNmJmMjg3MDdkNDAyZGQ3ZGQzNjg3ZTRlYzNlOGIzLzVkY2IxOTlkYmZiNWUxYjA0Njk5NmQwNzc5OGQyNmUxNTU1Y2IzZTg4MTYyOTkzODQzNzdhOWJjMzc3YWY0ZjY%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qJnJlc3BvbnNlLWNvbnRlbnQtdHlwZT0qIn1dfQ__&Signature=lnvLMCRaeGwhC5R8XppQnwTKa7IZHFsICX53DOE0GqGgySXTSZ1S0Fhfz-Hea09RG8TLGMeSA96ujBc2V73W2L%7ExPtTHObtBJNdqcGE08JlMDyB%7EEzz0ZbA78mGrxFrlgGsyK83rOhELfeB66kZTqDVTexIF8zaj-owz7scA6OSJP%7EZx72axstpnXojQiB4ivrug0KOIn9JcmmwnJUmF5PWSeyEfo0EMvgBBrgu7v49JSX3CSgNxRLLy759mN9eJwKU1vfHlBbn%7EsqyKXlG7-0WecmE%7EmpKH5ddedyOD9icW0ytf5MlfOnS5jGF-O69qgRg%7EkSOiYYEiqZ9xl5x7HA__&Key-Pair-Id=K3RPWS32NSSJCE"

        generated_image = generate_image_from_reference(prompt, image_url)

        # Convert generated image to base64
        buffered = BytesIO()
        generated_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_str = f"data:image/png;base64,{img_str}"

        print("Sending image...")
        return render_template('index.html', generated_image=img_str)

    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', error="Failed to generate image. Check server logs.")


if __name__ == '__main__':
    app.run(port=5000)
