from pyngrok import ngrok
from flask import Flask, jsonify, request, send_file
from dotenv import load_dotenv, find_dotenv
import os
import torch
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_gif
from io import BytesIO
from PIL import Image

# Load environment variables
load_dotenv(find_dotenv())
authtoken = os.getenv('NGROK_key')
ngrok.set_auth_token(authtoken)

# Initialize the pipeline on GPU
pipeline = I2VGenXLPipeline.from_pretrained(
    "ali-vilab/i2vgen-xl", 
    torch_dtype=torch.float16, 
    variant="fp16"
).to("cuda")  # Move pipeline to GPU

app = Flask(__name__)

@app.route('/generate_video', methods=['POST'])
def generate_video():
    try:
        image_file = request.files.get('image')
        prompt = request.form.get('prompt')
        negative_prompt = request.form.get('negative_prompt', '')

        if image_file is None or prompt is None:
            return jsonify({'error': 'Image and prompt are required'}), 400

        image = Image.open(image_file)
        frames = pipeline(prompt=prompt, image=image, negative_prompt=negative_prompt, num_frames=10).frames

        gif_bytes = export_to_gif(frames)
        gif_io = BytesIO(gif_bytes)
        gif_io.seek(0)

        return send_file(gif_io, mimetype='image/gif', as_attachment=True, download_name='generated_video.gif')

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print(f" * ngrok tunnel: {public_url}")

    app.run(port=5000)
