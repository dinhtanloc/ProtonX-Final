import streamlit as st
import requests
import uuid
from PIL import Image
from io import BytesIO

st.title("Multimodal Caption Generator")

# Generate a random session ID
session_id = str(uuid.uuid4())

# Initialize the session state for the backend URL
if "img_flask_api_url" not in st.session_state:
    st.session_state.img_flask_api_url = None

# Function to display the dialog and set the URL
@st.dialog("Setup Backend")
def setup_backend():
    st.markdown(
        """
        Paste the Ngrok link below.
        """
    )
    link = st.text_input("Backend URL", "")
    if st.button("Save"):
        st.session_state.img_flask_api_url = "{}/generate_video".format(link)  # Update ngrok URL
        st.rerun()  # Re-run the app to close the dialog

# Display the setup option if the URL is not set
if st.session_state.img_flask_api_url is None:
    setup_backend()

# Once the URL is set, display it or proceed with other functionality
if st.session_state.img_flask_api_url:
    st.success(f"Backend is set to: {st.session_state.img_flask_api_url}")

# TODO 1: Upload File
# uploaded_file = None
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Display the uploaded image with width 400
if uploaded_file is not None:
    st.image(uploaded_file, width=400)

# TODO 2: Get Prompt
user_prompt = st.text_input("Enter a prompt")

button = st.button("Generate Caption")

# Process inputs only if both are provided
if user_prompt and uploaded_file and button:
    # Prepare the data for the request
    files = {'image': uploaded_file}
    data = {'prompt': user_prompt}
    
    # TODO 3: Send promt and image to server
    # # response = None
    # response = requests.post(st.session_state.img_flask_api_url, files=files, data=data, timeout=600)
    # # if uploaded_file and user_prompt:
    # # if uploaded_file:
    # #     response = requests.post(st.session_state.img_flask_api_url, files=files, timeout=600)
    # # response = requests.post(st.session_state.img_flask_api_url, data=data, timeout=600)


    
    try:
        # Send the request to the backend
        response = requests.post(st.session_state.img_flask_api_url, files=files, data=data, timeout=600)

        # Check if the request was successful
        if response.status_code == 200:
            api_response = response.json()
            generated_caption = api_response.get("response", "No caption received.")
            
            # Load GIF data if it's present
            gif_data = BytesIO(response.content)
            gif_image = Image.open(gif_data)
            
            # Display caption and generated image
            st.markdown(f"### Generated Caption:")
            st.write(generated_caption)
            st.image(gif_image, width=400)
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Exception occurred: {str(e)}")

