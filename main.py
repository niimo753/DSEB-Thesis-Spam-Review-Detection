import time

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from src.preprocessing import load_data
from src.models import DistanceBasedClassifier
from src.model_selection import cross_validation

class FaceRecognitionModel:
    def __init__(self, preprocessor, classifier):
        self.preprocessor = preprocessor
        self.clf = classifier
        self.original_x_train, self.original_x_test, self.original_y_train, self.original_y_test = None, None, None, None
        self.x_train, self.x_test = None, None
        self.y_train, self.y_test = None, None
        self.y_pred = None

    def fit(self, X, y):
        # perform PCA
        if isinstance(self.preprocessor, LinearDiscriminantAnalysis):
            X_reduced = self.preprocessor.fit_transform(X, y)
        else:
            X_reduced = self.preprocessor.fit_transform(X)
        self.clf.fit(X_reduced, y)

        # store the original and reduced training data for potential future use
        self.original_x_train = X
        self.x_train = X_reduced
        self.y_train = y

    def predict(self, X):
        X_reduced = self.preprocessor.transform(X)
        self.original_x_test = X
        self.x_test = X_reduced
        self.y_pred = self.clf.predict(X_reduced)
        return self.y_pred

    def score(self, X, y):
        if self.y_pred is None:
            self.y_pred = self.predict(X)
        accuracy = accuracy_score(y, self.y_pred)
        return accuracy

    def get_params(self, deep=True):
        return {"preprocessor": self.preprocessor,
                "classifier": self.clf
                }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
    
    def get_projected_data(self):
        return self.x_train, self.x_test

if __name__ == '__main__':
    t0 = time.time()

    # Load data
    images, labels = load_data()

    # Initialize model
    model = FaceRecognitionModel(
        preprocessor=PCA(n_components=30),
        classifier=DistanceBasedClassifier(method="euclidean"),
    )
    print(model.get_params())

    # Cross validation accuracy
    scores = cross_validation(estimator=model, X=images, y=labels)
    
    # Round to 3 decimal places
    print('Accuracy:', scores, 'Mean accuracy:', round(scores.mean(), 3))

    preprocessor=PCA(n_components=30)
    print(preprocessor)

    print('Time taken:', time.time() - t0)