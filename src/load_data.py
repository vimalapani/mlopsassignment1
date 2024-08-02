
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_data():
    """Load the iris dataset and split it into train and test sets."""
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
