# Clases para los modelos de ML
import numpy as np

class LinearRegression():
    def __init__(self, X, y):
        self.feature_names = list(X.columns)
        self.X = X.values
        self.y = y.reshape(-1, 1)

        self.coef_inv = None
        self.coef_grad = None

        self.add_bias()

        print(self.X.shape)
        print(self.y.shape)
        print(len(self.feature_names))

    def add_bias(self):
        n_samples = self.X.shape[0]
        bias = np.ones((n_samples,1))

        self.X = np.hstack((bias, self.X))
        self.feature_names = ["bias"] + self.feature_names

    def pseudo_inverse(self):
        #entrenamiento por pseudo-inversa
        pinv = np.linalg.pinv(self.X.T @ self.X)
        w = pinv @ self.X.T @ self.y
        self.coef_inv = w

        print(w.shape)

    def mse(self, coef):
        if coef is None:
            raise ValueError("Modelo no entrenado aun")

        preds = self.X @ coef
        err = preds - self.y
        res = np.mean(err**2)
        return res
    
    def grad_mse(self, coef):
        preds = self.X @ coef
        err = preds - self.y
        grad = (2/self.X.shape[0]) * (self.X.T @ err)
        return grad

    def gradient_descent(self, lr=0.05, epochs=200):
        w = np.zeros((len(self.feature_names), 1))
        errors = []

        for i in range(epochs):
            grad = self.grad_mse(w)
            w = w - lr * grad

            errors.append(self.mse(w))

        self.coef_grad = w

        return errors

    def show_coef_inv(self):
        for feature, w in zip(self.feature_names, self.coef_inv):
            print(f"{feature}: {w[0]:.2f}")

    def show_coef_grad(self):
        for feature, w in zip(self.feature_names, self.coef_grad):
            print(f"{feature}: {w[0]:.2f}")