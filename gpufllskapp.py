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

# Ask user for ngrok authentication token
ngrok_token = input("Enter your ngrok authentication token: ")
ngrok.set_auth_token(ngrok_token)  # Set user-provided token
public_url = ngrok.connect(5000).public_url
print(f"Access your application at: {public_url}")

# Load the ControlNet model
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=dtype
).to(device)

# Load the Stable Diffusion pipeline with ControlNet
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=dtype
).to(device)


def load_image_from_url(url):
    """Download image from a URL and convert it to RGB."""
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


def generate_image_from_reference(prompt, image):
    """Generate an image based on a reference image and user prompt."""
    output = pipeline(
        prompt=prompt,
        image=image,
        negative_prompt="blurry, low quality, distorted",
    )
    return output.images[0]


@app.route('/')
def initial():
    return render_template('index.html')


@app.route('/submit-prompt', methods=['POST'])
def generate_image():
    try:
        prompt = request.form['prompt-input']
        image_url = request.form['image-url']

        print(f"Generating an image for: {prompt}")
        print(f"Using reference image from: {image_url}")

        input_image = load_image_from_url(image_url)  # Load user-provided image URL
        generated_image = generate_image_from_reference(prompt, input_image)

        # Convert generated image to base64
        buffered = BytesIO()
        generated_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_str = f"data:image/png;base64,{img_str}"

        print("Sending generated image to frontend...")
        return render_template('index.html', generated_image=img_str)

    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', error="Failed to generate image. Check server logs.")


if __name__ == '__main__':
    app.run(port=5000)
