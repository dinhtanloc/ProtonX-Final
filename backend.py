from pyngrok import ngrok
from flask import Flask, jsonify, request
from flask_cors import CORS
# from google.colab import userdata
import json
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
# Replace 'YOUR_NGROK_AUTHTOKEN' with the authtoken you copied from the ngrok dashboard
authtoken =os.getenv('NGROK_key')
ngrok.set_auth_token(authtoken)
import os
import torch
from flask import Flask, request, jsonify, send_file
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_gif, load_image
from pyngrok import ngrok
from io import BytesIO
from PIL import Image

# Initialize the pipeline
pipeline = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16")
pipeline.enable_model_cpu_offload()

app = Flask(__name__)

# Define route to receive image and prompt, and return a video (GIF)
@app.route('/generate_video', methods=['POST'])
def generate_video():
    try:
        # TODO 1: Get the image and prompt from the client
        # image_file = None
        # prompt = None
        # negative_prompt = None
        image_file = request.files.get('image')
        prompt = request.form.get('prompt')
        negative_prompt = request.form.get('negative_prompt', '')

        if image_file is None or prompt is None:
            return jsonify({'error': 'Image and prompt are required'}), 400

        # TODO 2: Load the image
        # image = None
        image = Image.open(image_file)

        # TODO 3: Generate frames
        # frames = None
        frames = pipeline(prompt=prompt, image=image, negative_prompt=negative_prompt, num_frames=10).frames

        # TODO 4: Send to client

        # return None
        gif_bytes = export_to_gif(frames)
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
