# Iris Classification Project

This is a simple machine learning project that trains a logistic regression model to classify iris flowers based on their sepal and petal measurements. The project uses the iris dataset from scikit-learn.

## Project Structure

The project has the following structure:

iris-classification/

├── load_data.py

├── train.py

├── deploy.py

├── tests/

│ └── test_model.py

├── requirements.txt

├── model.joblib

└── README.md


- `load_data.py`: Contains a function to load the iris dataset and split it into train and test sets.
- `train.py`: Contains a function to train a logistic regression model on the iris dataset and save the trained model to disk.
- `deploy.py`: Contains a function to load the trained model and make predictions.
- `tests/test_model.py`: Contains basic tests for the training and prediction processes.
- `requirements.txt`: Lists the Python dependencies for the project.
- `model.joblib`: The trained logistic regression model (generated after running `train.py`).
- `README.md`: This file, providing an overview of the project.

## Getting Started

# 1. Clone the repository:

git clone https://github.com/vimalapani/mlopsassignment1


# 2. Install the required dependencies:

pip install -r requirements.txt


# 3. Train the model:

python train.py


This will train the logistic regression model on the iris dataset and save the trained model to `model.joblib`.

# 4. Make predictions:

from deploy import predict
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
predictions = predict(X)


The predict function from deploy.py loads the trained model and makes predictions on the provided data.

# Run tests:

pytest tests/

This will run the basic tests for the training and prediction processes.
