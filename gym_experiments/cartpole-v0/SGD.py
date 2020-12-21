from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import NotFittedError
import numpy as np



class sklearnSGD:

    def __init__(self, step_size, num_actions, feature_size, random_state = 0):
        self.regressors = [SGDRegressor(eta0=step_size, random_state=random_state)
                                for _ in range(num_actions)]
        
        self.num_actions = num_actions
        self.feature_size = feature_size
        self.default_value = 0
        self.f_vec = np.zeros(feature_size)

    def update(self, tiles, y, index):
        self.f_vec[tiles] = 1
        x = self.f_vec.reshape(1, -1)
        self.regressors[index].partial_fit(x, y)
        self.f_vec[tiles] = 0

    @property
    def weights(self):
        reg_weights = list()
        for i in range(self.num_actions):
            if not hasattr(self.regressors[i], "coef_"):
                reg_weights.append(np.array([0.0] * self.feature_size))
            else:
                reg_weights.append(self.regressors[i].coef_)
        return np.vstack(reg_weights)


    def run(self, tiles, index):
        self.f_vec[tiles] = 1
        try:
            if self.f_vec.ndim == 1:
                x = np.expand_dims(self.f_vec, axis=0)
            res = self.regressors[index].predict(x)[0]  # TODO: Doesn't work for batch inputs yet!!!
        except NotFittedError:
            print(f"Regressor for action {index} not fitted yet. Returning default value {self.default_value}.")
            res = self.default_value
        self.f_vec[tiles] = 0
        return res




class numpySGD:
    pass
    

            
def sgd_factory(library):
    sgd_class_mapping = {"sklearn": sklearnSGD}
    return sgd_class_mapping[library]



# n_samples, n_features = 10, 5
# rng = np.random.RandomState(0)
# y = rng.randn(n_samples)
# X = rng.randn(n_samples, n_features)
# learning_rate = 0.01
# expected_weights = np.array([-0.05369174, 0.02041468, -0.02989519, 0.01707805, 0.04550364])
# sgd = sgd_factory("sklearn")
# model = sgd(learning_rate, 2, 5)
# model.update(X, y, 1)
# model.update(X - 1, y + 1, 0)
# print(model.weights)