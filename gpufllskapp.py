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





from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request
import torch
import base64
from io import BytesIO
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import OpenposeDetector
from diffusers import UniPCMultistepScheduler
from PIL import Image
from IPython.display import display
from pyngrok import ngrok

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32  # Adjust dtype

# Initialize Flask app
app = Flask(__name__)

# ngrok setup for public URL
ngrok.set_auth_token("2jK44YcMq9wfqWna2tpf4gkZxCY_24fLpcRmNfyZqQQhcyJ5M")
public_url = ngrok.connect(5000).public_url

print(f"To access the global link, please click {public_url}")

def generate_images(generate_prompt, model_id="runwayml/stable-diffusion-v1-5"):
    # Use a placeholder input image for pose detection
    input_image = Image.new("RGB", (512, 512), (255, 255, 255))  # Dummy white image

    # Load Openpose model for property extraction
    pose_model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    pose_image = pose_model(input_image)

    # Load ControlNet model
    controlnet = ControlNetModel.from_pretrained(
        "fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float16
    )

    # Load Stable Diffusion pipeline with ControlNet
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    # Generate new images
    generator = torch.Generator(device="cpu").manual_seed(2)
    output = pipe(
        [generate_prompt] * 4,  # Generate multiple images
        [pose_image] * 4,
        generator=[torch.Generator(device="cpu").manual_seed(i) for i in range(4)],
        num_inference_steps=20,
    )

    images = output.images  # Store all images in a variable
    return images  # Return all generated images

@app.route('/')
def initial():
    return render_template('index.html')

@app.route('/submit-prompt', methods=['POST'])
def generate_image():
    try:
        # Get the prompt submitted from the form
        prompt = request.form['prompt-input']
        print(f"Generating an image for: {prompt}")

        # Generate images using the provided prompt
        images = generate_images(prompt)
        print("Images generated! Converting to base64...")

        # Convert the first generated image to base64
        buffered = BytesIO()
        images[0].save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_str = f"data:image/png;base64,{img_str}"

        print("Sending image...")
        return render_template('index.html', generated_image=img_str)

    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', error="Failed to generate image. Check server logs.")

if __name__ == '__main__':
    app.run(port=5000)
