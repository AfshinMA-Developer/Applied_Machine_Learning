# Import required libraries
import os
import keras
import requests
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Function to safely load the models
def load_model_safely(path: str):
    if not os.path.isfile(path) or not path.endswith('.keras'):
        raise FileNotFoundError(f"The file '{path}' does not exist or is not a .keras file.")
    return keras.saving.load_model(path)

# Retrieve the current directory and specify model paths
current_dir = os.getcwd()  # Ensure correct initial directory
model_paths = {
    'CNN': os.path.join(current_dir, 'models', 'cnn_model.keras'),
    'VGG19': os.path.join(current_dir, 'models', 'vgg19_model.keras'),
    'ResNet50': os.path.join(current_dir, 'models', 'resnet50_model.keras'),
}
model_urls = {
    'CNN': 'https://drive.google.com/file/d/1HKWSxMj7odYihQnO8gy_Do2VXc2uT1vn/view?usp=drive_link',
    'VGG19': 'https://drive.google.com/file/d/1z20Z9ZLVg-693aIZaUd1XtXq32MzyOnV/view?usp=drive_link',
    'ResNet50': 'https://drive.google.com/file/d/1LxY39-rb0UFeRp-NTAsN-vUeY9s_wqh9/view?usp=drive_link',
}

# Load models and handle potential exceptions
models = {}
for name, path in model_paths.items():
    try:
        if not os.path.exists(path):
            url = model_urls[name]
            path = model_paths[name]

            response = requests.get(url)
            
            if response.status_code == 200:
                with open(path, 'wb') as file:
                    file.write(response.content)
                print(f'> {name} model downloaded successfully.')
            else:
                models[name] = None
                print(f'> Failed to download {name}model ...!')

        models[name] = load_model_safely(path)   
        
    except Exception as e:
        st.error(f"Error loading model {name} from {path}: {str(e)}")

# Define the class labels
classes = { 0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)', 
            9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection', 
            12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution', 
            19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve', 
            22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right', 
            25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing', 
            29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing', 
            32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead', 
            35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left', 
            38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory', 
            41:'End of no passing', 42:'End no passing veh > 3.5 tons' }

# Function to preprocess the image and predict the class
def preprocess_and_predict(image: Image.Image, size=(50, 50)) -> pd.DataFrame:
    img_resized = image.resize(size)
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape (1, 50, 50, 3)

    predictions = []
    for name, model in models.items():
        predicted_class_index = np.argmax(model.predict(img_array), axis=-1)[0]
        predictions.append({'Model': name, 'Predicted Label': classes[predicted_class_index]})

    return pd.DataFrame(predictions)

# Import Example images
images_dir = os.path.join(current_dir, 'images')

if os.path.exists(images_dir):
    # Create a list of images and their corresponding classes
    image_list = [img for img in os.listdir(images_dir) if img.lower().endswith('.png')]
    image_dict = {classes[int(img.split('.')[0])] : os.path.join(images_dir, img) for img in image_list}
else:
    st.error(f"The images directory does not exist: {images_dir}")

# Streamlit UI setup
st.set_page_config(page_title="Traffic Sign Detection App", page_icon="🚦", layout="wide")
st.title("🚦 Traffic Sign Recognition using CNN, VGG19, ResNet50")
st.markdown("Upload a traffic sign image or choose an example from below to get the recognition result.")
st.markdown("---")

# Sidebar for image upload and selection
st.sidebar.header("Input Options")
uploaded_file = st.sidebar.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

# Select an example image
selected_example = st.sidebar.selectbox("Or select an example image:", list(image_dict.keys()))
if selected_example:
    example_image_path = image_dict[selected_example]

# Initialize a variable to hold the image for prediction
image_to_predict = None

# Check if user uploaded an image or selected an example image
if uploaded_file is not None:
    image_to_predict = Image.open(uploaded_file)
    st.image(image_to_predict.resize((256, 256)), caption='Uploaded Image', use_container_width=False, output_format="auto")
elif selected_example:
    image_to_predict = Image.open(example_image_path)
    st.image(image_to_predict.resize((256, 256)), caption='Example Image', use_container_width=False, output_format="auto")

# Add a predict button
if st.sidebar.button("🚀 Predict", key="predict_button") and image_to_predict is not None:
    # Run prediction
    st.write("Predicting ...")
    results = preprocess_and_predict(image_to_predict)
    
    # Display results
    st.write("### Prediction Results")

    # Style the output dataframe
    st.dataframe(results)

# Add some custom CSS for better styling
st.markdown("""
<style>
    .stButton > button:hover {
        background-color: #0052cc; /* Darker blue on hover */
    }
    .stDataframe {
        border: 1px solid #ddd;     /* Light border for clarity */
        border-radius: 10px;        /* Rounded corners for the dataframe */
    }
    .stImage {
        border: 2px solid #0066ff; /* Border for images */
        border-radius: 10px;       /* Rounded corners */
        box-shadow: 0 0 8px rgba(0, 0, 0, 0.2); /* Subtle shadow */
    }
</style>
""", unsafe_allow_html=True)
