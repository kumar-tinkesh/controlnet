from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request
import torch
import base64
from io import BytesIO
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32  # Adjust dtype

# Load ControlNet model
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", 
    torch_dtype=dtype  # Use adjusted dtype
)

# Load Stable Diffusion with ControlNet
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controlnet, 
    torch_dtype=dtype  # Use adjusted dtype
)

pipe.to(device)


# Start Flask app
app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def initial():
    return render_template('index.html')

@app.route('/submit-prompt', methods=['POST'])
def generate_image():
    try:
        prompt = request.form['prompt-input']
        print(f"Generating an image of: {prompt}")

        # Generate an image
        image = pipe(prompt).images[0]
        print("Image generated! Converting to base64...")

        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_str = f"data:image/png;base64,{img_str}"

        print("Sending image...")
        return render_template('index.html', generated_image=img_str)

    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', error="Failed to generate image. Check server logs.")

if __name__ == '__main__':
    app.run()
