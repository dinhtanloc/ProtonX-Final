from pyngrok import ngrok
from flask import Flask, jsonify, request, send_file
from PIL import Image
from io import BytesIO
import os
import torch
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_gif
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Replace 'YOUR_NGROK_AUTHTOKEN' with your actual ngrok token
authtoken = os.getenv('NGROK_key')
ngrok.set_auth_token(authtoken)

# Initialize the pipeline
pipeline = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16")
pipeline.enable_model_cpu_offload()

app = Flask(__name__)

# Define route to receive image and prompt, and return a video (GIF)
@app.route('/generate_video', methods=['POST'])
def generate_video():
    try:
        # Get the image and prompt from the client
        image_file = request.files.get('image')
        prompt = request.form.get('prompt')
        negative_prompt = request.form.get('negative_prompt', '')

        if image_file is None or prompt is None:
            return jsonify({'error': 'Image and prompt are required'}), 400

        # Load the image
        image = Image.open(image_file)

        # Generate frames
        frames = pipeline(prompt=prompt, image=image, negative_prompt=negative_prompt, num_frames=10).frames

        # Export frames to GIF
        gif_bytes = export_to_gif(frames)

        # Save the GIF locally
        local_path = "generated_video.gif"
        with open(local_path, "wb") as f:
            f.write(gif_bytes)

        # Prepare the GIF to be sent to the client
        gif_io = BytesIO(gif_bytes)
        gif_io.seek(0)

        return send_file(gif_io, mimetype='image/gif', as_attachment=True, download_name='generated_video.gif')

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Start Flask server and expose it via ngrok
    public_url = ngrok.connect(5000)
    print(f" * ngrok tunnel: {public_url}")

    app.run(port=5000)
