import gradio as gr
import pandas as pd
import joblib
from sklearn.metrics import r2_score
from utils import data_cleaning

# Load Dataset and Models
DATASET_PATH = "datasets/housePrice.csv"
MODELS_PATH = [
    "models/KernelRidge_pipeline.joblib",
    "models/GradientBoostingRegressor_pipeline.joblib",
    "models/XGBoostRegressor_pipeline.joblib",
    "models/CatBoostRegressor_pipeline.joblib",
]

# Load the cleaned data
df = data_cleaning(DATASET_PATH)

# Prepare data for input fields
min_area, max_area = df['Area'].min(), df['Area'].max()
rooms = df['Room'].unique().tolist()
addresses = df['Address'].unique().tolist()

def load_and_predict(area, room, parking, warehouse, elevator, address):
    sample = pd.DataFrame({
        'Area': [area],
        'Room': [room],
        'Parking': [parking],
        'Warehouse': [warehouse],
        'Elevator': [elevator],
        'Address': [address]
    })

    result = {
        'Model': [],
        'R2': [],
        'Predicted_Price_(IRR)': []
    }

    # Define features and target variable
    X = df.drop(columns=['Price'])  # Features
    y = df['Price']

    try:
        for path in MODELS_PATH:
            model_name = path.split('/')[-1].split('_')[0]
            model = joblib.load(path)  # Load the model once

            # Predict house price
            y_pred = model.predict(X)
            price_pred = model.predict(sample)[0]

            result['Model'].append(model_name)
            result['R2'].append(r2_score(y, y_pred))
            result['Predicted_Price_(IRR)'].append(price_pred)

    except Exception as e:
        return f"An error occurred during model loading or prediction: {str(e)}"

    return pd.DataFrame(result).sort_values(by=['R2'], ascending=False)

# Create Gradio interface
iface = gr.Interface(
    fn=load_and_predict,
    inputs=[
        gr.Number(label="Area (m²)", value=min_area, minimum=min_area, maximum=max_area),
        gr.Dropdown(choices=rooms, label="Room", value=rooms[0]),
        gr.Checkbox(label="Parking", value=True),
        gr.Checkbox(label="Warehouse", value=True),
        gr.Checkbox(label="Elevator", value=True),
        gr.Dropdown(choices=addresses, label="Address", value=addresses[0])
    ],
    outputs="dataframe",
    title="🏠 Tehran House Price Prediction",
    description="This app predicts house prices based on input features such as area, number of rooms, and facilities like parking, warehouse, and elevator. Please fill in all fields to get the prediction."
)

# Launch the interface
iface.launch()