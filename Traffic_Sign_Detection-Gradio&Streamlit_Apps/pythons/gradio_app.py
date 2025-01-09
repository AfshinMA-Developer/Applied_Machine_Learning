# Import required libraries
import os
import keras
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

# Function to safely load the models
def load_model_safely(path: str):
    if not os.path.isfile(path) or not path.endswith('.keras'):
        raise FileNotFoundError(f"The file '{path}' does not exist or is not a .keras file.")
    return keras.saving.load_model(path)

# Retrieve the current directory and specify model paths
current_dir = os.getcwd()  # Ensure correct initial directory
model_paths = {
    'CNN': os.path.join(current_dir, 'Project_7_Traffic_Sign_Detection', 'models', 'cnn_model.keras'),
    'VGG19': os.path.join(current_dir, 'Project_7_Traffic_Sign_Detection', 'models', 'vgg19_model.keras'),
    'ResNet50': os.path.join(current_dir, 'Project_7_Traffic_Sign_Detection', 'models', 'resnet50_model.keras'),
}

# Load models and handle potential exceptions
models = {}
for name, path in model_paths.items():
    try:
        models[name] = load_model_safely(path)
    except Exception as e:
        print(f"Error loading model {name} from {path}: {str(e)}")

# Define the class labels
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Veh > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing veh > 3.5 tons' }

# Function to import and resize example images
def get_example_images(images_dir:str, size=(50, 50)) -> list:

    # Check if the images directory exists
    if not os.path.exists(images_dir):
        print(f"The images directory does not exist: {images_dir}")
        return []
    
    images = []
    image_list = os.listdir(images_dir)
    for image in image_list:
        if image.lower().endswith('.png'):
            image_path = os.path.join(images_dir, image)
            img = Image.open(image_path)
            img = img.resize(size)
            images.append(img)
    return images

# Directory for example images
images_dir = os.path.join(current_dir, 'Project_7_Traffic_Sign_Detection', 'images')
examples = get_example_images(images_dir, (50, 50))

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

# Create Gradio interface
iface = gr.Interface(
    fn=preprocess_and_predict,
    inputs=gr.Image(type='pil'),  # Changed to 'pil' for direct use with PIL
    outputs="dataframe",  # Correct the output type
    examples=examples,
    title="Traffic Sign Recognition",
    description="Upload a traffic sign image or choose an example to get the recognition result."
)

# Launch the Gradio app
iface.launch()