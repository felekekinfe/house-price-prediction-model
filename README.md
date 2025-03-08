
# House Price Prediction Model

![Python](https://img.shields.io/badge/Python-3.x-blue.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

Welcome to the **House Price Prediction Model**! This project uses **Linear Regression** to predict house prices in the USA based on key features like income, house age, number of rooms, bedrooms, and population. The model is trained on real-world data and saved for easy reuse.

## Features
- Predicts house prices using:
  - **Average Area Income**
  - **Average Area House Age**
  - **Average Number of Rooms**
  - **Average Number of Bedrooms**
  - **Area Population**
- Achieves ~91.7% accuracy on test data.
- Trained model saved in `.pkl` format for future predictions.

## Project Structure
```
house-price-prediction-model/
├── dataset/
│   └── USA_Housing.csv         
├── model/
│   └── _model.pkl             
├── main/
│   └── prediction_model.py   
├── README.md                 
└── requirements.txt          
```

## Prerequisites
- **Python 3.x**
- Required libraries:
  - `pandas` (data manipulation)
  - `scikit-learn` (machine learning)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/felekekinfe/house-price-prediction-model.git
   cd house-price-prediction-model
   ```
2. (Optional) Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
The dataset (`USA_Housing.csv`) contains housing data with the following columns:

| Column Name                | Description                          |
|----------------------------|--------------------------------------|
| Avg. Area Income           | Average income in the area           |
| Avg. Area House Age        | Average age of houses                |
| Avg. Area Number of Rooms  | Average number of rooms              |
| Avg. Area Number of Bedrooms | Average number of bedrooms         |
| Area Population            | Population of the area               |
| Price                      | House price (target variable)        |

## Training the Model
1. Place `USA_Housing.csv` in the `dataset/` folder.
2. Run the training script:
   ```bash
   python main/prediction_model.py
   ```
3. Output example:
   ```
   X shape: (5000, 5)
   Y shape: (5000,)
   x_train shape: (4000, 5)
   y_train shape: (4000,)
   Model training and saving completed successfully.
   ```
   The trained model is saved as `_model.pkl` in the `model/` folder.

## Model Performance
- **Accuracy**: ~91.7% on test data
- Built using **Linear Regression** from `scikit-learn`.

## Making Predictions
Use the trained model to predict house prices with new data:

```python
import pickle
import numpy as np

# Load the model
with open('model/_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Sample input: [income, age, rooms, bedrooms, population]
sample_data = np.array([[65000, 10, 6, 3, 50000]])

# Predict
predicted_price = model.predict(sample_data)
print(f"Predicted house price: ${predicted_price[0]:,.2f}")
