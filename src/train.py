
from sklearn.linear_model import LogisticRegression
from load_data import load_data
import joblib

def train_model():
    """Train a logistic regression model on the iris dataset."""
    X_train, X_test, y_train, y_test = load_data()
    model = LogisticRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Test accuracy: {score:.2f}")
    joblib.dump(model, "model.joblib")


if __name__ == "__main__":
    train_model()
