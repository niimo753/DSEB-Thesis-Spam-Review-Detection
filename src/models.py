import numpy as np
from sklearn.metrics import accuracy_score

class DistanceBasedClassifier:
    def __init__(self, method='euclidean', norm=None, method_type="normal", normed_vector=False):
        self.method = method
        self.X_train = None
        self.y_train = None
        self.norm = norm
        self.method_type = method_type
        self.normed_vector = normed_vector
        self.distance = None
        self.pred_index = None
        self.y_pred = None

    def __repr__(self):
        params = self.get_params()
        no_print = [None, "normal", False]
        filter_params = {key: value for key, value in params.items() if value not in no_print}
        param_str = ', '.join(f"{key}={value}" for key, value in filter_params.items())
        return f"DistanceBasedClassifier({param_str})"
    
    def get_params(self, deep=True):
        return {'method': self.method,
                'norm': self.norm,
                'method_type': self.method_type,
                'normed_vector': self.normed_vector
                }

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = []
        dists = []
        pred_index = []
        if self.normed_vector:
            X_test = self._normalize_vector(X)
            self.X_train = self._normalize_vector(self.X_train)
        else:
            X_test = X
        for item in X_test:
            item_dists = self._predict_one(item)
            dists.append(item_dists)
            item_index = np.argmin(item_dists)
            pred_index.append(item_index) 
            y_pred.append(self.y_train[item_index])
        self.distance = np.array(dists)
        self.pred_index = np.array(pred_index)
        self.y_pred = np.array(y_pred)
        return self.y_pred

    def _predict_one(self, x_test):
        methods = {"euclidean": self._euclidean,
                   "manhattan": {"normal": self._manhattan,
                                 "weighted": self._weighted_manhattan,
                                 "modified": self._modified_manhattan,
                                 "weighted-modified": self._weighted_modified_manhattan},
                   "minkowski": self._minkowski,
                   "sse": {"normal": self._sse,
                           "weighted": self._weighted_sse,
                           "modified": self._modified_sse,
                           "weighted-modified": self._weighted_modified_sse},
                   "mse": self._mse,
                   "angle": {"normal": self._angle,
                             "weighted": self._weighted_angle},
                   "correlation-coefficient": self._corr_coeff,
                   "mahalanobis": {"normal": self._mahalanobis,
                                   "using-eigenvalues": self._mahalanobis_eigvals},
                   "chi-square": self._chi_square,
                   "canberra": self._canberra}
        if self.method not in methods.keys():
            raise Warning("Invalid input method. Available methods for distance based classification: {}".format(", ".join(methods.keys())))
        else:
            if isinstance(methods[self.method], dict):
                if self.method_type not in methods[self.method].keys():
                    raise Warning("Invalid input type for {} method. Available input types: {}".format(self.method, ", ".join(methods[self.method])))
                else:
                    return methods[self.method][self.method_type](x_test)
            else:
                return methods[self.method](x_test)

    def _normalize_vector(self, vector):
        magnitude = np.sqrt(np.sum(vector**2))
        return vector/magnitude

    def _minkowski(self, x_test):
        dists = np.linalg.norm(self.X_train - x_test, ord=self.norm, axis=1)
        return dists
    
    def _euclidean(self, x_test):
        dists = np.linalg.norm(self.X_train - x_test, ord=2, axis=1)
        return dists
    
    def _manhattan(self, x_test):
        dists = np.sum(np.abs(self.X_train - x_test), axis=1)
        return dists
    
    def _sse(self, x_test):
        dists = np.sum((self.X_train - x_test)**2, axis=1)
        return dists

    def _mse(self, x_test):
        dists = np.mean((self.X_train - x_test)**2, axis=1)
        return dists
    
    def _angle(self, x_test):
        dists = []
        for stored_item in self.X_train:
            distance = np.dot(stored_item, x_test)
            divisor = np.sum(x_test**2)*np.sum(stored_item**2)
            distance = - distance/np.sqrt(divisor)
            dists.append(distance)
        return np.array(dists)

    def _corr_coeff(self, x_test):
        dists = []
        n = x_test.shape[0]
        for stored_item in self.X_train:
            distance = n*np.dot(stored_item, x_test) - np.sum(stored_item)*np.sum(x_test)
            divisor = np.sqrt((n*np.sum(stored_item**2) - np.sum(stored_item)**2)*(n*np.sum(x_test**2) - np.sum(x_test)**2))
            distance = - distance/divisor
            dists.append(distance)
        return np.array(dists)
    
    def _calculate_z(self, simplified=True):
        cov = np.cov(self.X_train, rowvar=0)
        eigval = np.linalg.eigvals(cov)
        z = abs(1 / eigval)
        return np.sqrt(z)
    
    def _mahalanobis(self, x_test):
        # general formula
        dists = []
        cov = np.cov(self.X_train, rowvar=0)
        covinv = np.linalg.inv(cov)
        for stored_item in self.X_train:
            a = stored_item - x_test
            distance = a.dot(covinv).dot(a)
            distance = abs(distance)
            distance = np.sqrt(distance)
            dists.append(distance)
        return np.array(dists)
    
    def _mahalanobis_eigvals(self, x_test):
        # formula in "Distance measures for PCA..." paper
        dists = []
        z = self._calculate_z(simplified=False)
        z = np.identity(n=z.shape[0])*z
        for stored_item in self.X_train:
            distance = z.dot(stored_item)
            distance = - distance.dot(x_test)
            dists.append(distance)
        return np.array(dists)

    def _weighted_manhattan(self, x_test):
        dists = []
        z = self._calculate_z()
        for stored_item in self.X_train:
            x = abs(x_test - stored_item)
            distance = np.dot(z, x)
            dists.append(distance)
        return np.array(dists)

    def _weighted_sse(self, x_test):
        dists = []
        z = self._calculate_z()
        for stored_item in self.X_train:
            x = (x_test - stored_item)**2
            distance = np.dot(z, x)
            dists.append(distance)
        return np.array(dists)

    def _weighted_angle(self, x_test):
        dists = []
        z = self._calculate_z()
        for stored_item in self.X_train:
            z = np.identity(n=z.shape[0])*z
            distance = z.dot(stored_item)
            distance = distance.dot(x_test)
            divisor = np.sqrt(np.sum(stored_item**2)*np.sum(x_test**2))
            distance = - distance/divisor
            dists.append(distance)
        return np.array(dists)

    def _chi_square(self, x_test):
        dists = []
        for stored_item in self.X_train:
            distance = np.sum((x_test - stored_item)**2 / (x_test + stored_item))
            dists.append(distance)
        return np.array(dists)

    def _canberra(self, x_test):
        dists = np.sum(np.abs(x_test - self.X_train) / (np.abs(x_test) + np.abs(self.X_train)), axis=1)
        return dists

    def _modified_manhattan(self, x_test):
        dists = []
        for stored_item in self.X_train:
            distance = np.sum(np.abs(x_test - stored_item)) / (np.sum(np.abs(x_test)) * np.sum(np.abs(stored_item)))
            dists.append(distance)
        return np.array(dists)

    def _modified_sse(self, x_test):
        dists = []
        for stored_item in self.X_train:
            distance = np.sum((x_test - stored_item)**2) / (np.sum(x_test**2) * np.sum(stored_item**2))
            dists.append(distance)
        return np.array(dists)

    def _weighted_modified_manhattan(self, x_test):
        dists = []
        weights = self._calculate_z()
        for stored_item in self.X_train:
            distance = weights.dot(np.abs(x_test - stored_item)) / (np.sum(np.abs(x_test)) * np.sum(np.abs(stored_item)))
            dists.append(distance)
        return np.array(dists)

    def _weighted_modified_sse(self, x_test):
        dists = []
        weights = self._calculate_z()
        for stored_item in self.X_train:
            distance = weights.dot((x_test - stored_item)**2) / (np.sum(x_test**2) * np.sum(stored_item**2))
            dists.append(distance)
        return np.array(dists)
    
    def get_predicted_index(self):
        return self.pred_index
    
    def get_distance(self):
        return self.distance
    
    def kneighbors(self, X=None, n_neighbors=None, return_distance=False):
        self.predict(X)
        distance = self.get_distance()
        sorted_index = np.argsort(distance, axis=1)
        k_index = np.array([i[:n_neighbors] for i in sorted_index])
        k_distance = np.array([i[k] for i, k in zip(distance, k_index)])
        if return_distance:
            return k_distance, k_index
        else:
            return k_index
        
    def score(self, X, y):
        if self.y_pred is None:
            self.y_pred = self.predict(X)
        accuracy = accuracy_score(y, self.y_pred)
        return accuracy

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self