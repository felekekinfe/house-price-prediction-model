
House Price Prediction Model

Welcome to the House Price Prediction project! This project uses Linear Regression to predict house prices based on various factors such as:

    Average Area Income
    Average Area House Age
    Average Number of Rooms
    Average Number of Bedrooms
    Area Population

The model is trained on a real-world dataset of housing prices in the USA, and the trained model is saved for future use.
📂 Project Structure

The project structure looks like this:

house-price-prediction-model/
│
├── dataset/
│   └── USA_Housing.csv         
│
├── model/
│   └── _model.pkl               
│
├── main/
│   └── prediction_model.py      
│
├── README.md                   
└── requirements.txt             
🛠 Requirements

To run the project, you'll need Python 3.x and the following dependencies:

    pandas: For data manipulation and analysis
    scikit-learn: For building and training the machine learning model

1. Set up the environment

Step 1: Clone this repository:

git clone https://github.com/felekekinfe/house-price-prediction-model.git
cd house-price-prediction-model

Step 2: Create a virtual environment (optional but recommended):

python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Step 3: Install the required libraries:

pip install -r requirements.txt

You can generate the requirements.txt file by running:

pip freeze > requirements.txt

📊 Dataset

The dataset used in this project is USA_Housing.csv, which contains information about various houses in the USA. Here are the key columns:
Column Name	Description
Avg. Area Income	The average income of the area.
Avg. Area House Age	The average age of houses in the area.
Avg. Area Number of Rooms	The average number of rooms in the area.
Avg. Area Number of Bedrooms	The average number of bedrooms in the area.
Area Population	The population of the area.
Price	The target variable (house price).
📝 How to Train the Model

To train the model, simply run the prediction_model.py script. This script will:

    Load the dataset.
    Split the data into training and testing sets.
    Train a Linear Regression model.
    Save the trained model as a .pkl file.

Steps:

    Ensure the dataset (USA_Housing.csv) is placed inside the dataset/ folder.
    Run the following command:

python main/prediction_model.py

Example Output:

X shape: (5000, 5)
Y shape: (5000,)
x_train shape: (4000, 5)
y_train shape: (4000,)
Model training and saving completed successfully.

The model will be saved as _model.pkl inside the model/ directory.
📈 Model Accuracy

The Linear Regression model achieves an accuracy of approximately 91.7% on the test data. This indicates that the model is able to make reasonably accurate predictions about house prices.
🤖 Using the Trained Model for Predictions

Once the model is trained and saved, you can use it to make predictions on new data.
Example Usage:

import pickle
import numpy as np

# Load the trained model
with open('model/_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Sample data (replace with your own data)
sample_data = np.array([[65000, 10, 6, 3, 50000]])

# Predict house price
predicted_price = model.predict(sample_data)
print(f"Predicted house price: ${predicted_price[0]:,.2f}")

This code loads the saved model and uses it to predict the price of a house based on sample input data.
💡 Contributing
