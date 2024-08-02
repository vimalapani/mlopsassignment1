
import joblib

def predict(X):
    """Make predictions using the trained model."""
    model = joblib.load("model.joblib")
    return model.predict(X)