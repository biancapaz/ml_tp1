# Clases para los modelos de ML
import numpy as np
from metrics import mse

class LinearRegression():
    def __init__(self, X, y, l2=0, l1=0):

        self.feature_names = list(X.columns)
        self.X = X.values
        self.y = y.reshape(-1, 1)

        # regularization coefs L1 and L2
        self.l2 = l2
        self.l1 = l1

        # coefs depending on training method
        self.coef_inv = None
        self.coef_grad = None

        # add bias to X matrix before anything
        self.add_bias()

    def add_bias(self):

        n_samples = self.X.shape[0]
        bias = np.ones((n_samples, 1))

        self.X = np.hstack((bias, self.X))
        self.feature_names = ["bias"] + self.feature_names

    def pseudo_inverse(self):
        #entrenamiento por pseudo-inversa

        xtx = self.X.T @ self.X

        if self.l2 < 0:
            print("entra < 0")
            pinv = np.linalg.pinv(xtx)
        else:
            print("entra else")
            ident = np.identity(self.X.shape[1])
            ident[0,0] = 0 # pongo en 0 para que no regularice bias

            pinv = np.linalg.pinv(xtx + self.l2 * ident)
            
        w = pinv @ self.X.T @ self.y

        self.coef_inv = w
    
    def grad_mse(self, coef):
        # necesario para gradient_descent

        preds = self.X @ coef
        err = preds - self.y
        grad = (2/self.X.shape[0]) * (self.X.T @ err)

        # si hacemos con regulacion cambia el gradiente!!
        if self.l1 != 0:
            reg = self.l1 * np.sign(coef)
            reg[0] = 0 # no regularizo el bias
            grad += reg

        return grad

    def gradient_descent(self, lr=0.001, epochs=1000):

        w = np.zeros((self.X.shape[1], 1))
        errors = []

        for _ in range(epochs):
            grad = self.grad_mse(w)

            w = w - lr * grad

            y_pred = self.X @ w
            errors.append(mse(self.y, y_pred))

        self.coef_grad = w

        return errors
    
    """
    def predict(self, X_val):

        # convierto a np array
        X_val_np = X_val.values

        # agregar bias
        bias = np.ones((X_val_np.shape[0], 1))
        X_val_np = np.hstack((bias, X_val_np))
        
        if self.coef_grad is not None:
            return X_val_np @ self.coef_grad
        elif self.coef_inv is not None:
            return X_val_np @ self.coef_inv
        else:
            raise ValueError("Modelo no entrenado")
    """
    
    def predict_inv(self, X_val):
        if self.coef_inv is None:
            raise ValueError("Modelo no entrenado con pseudo-inversa")

        # convierto a np array
        X_val_np = X_val.values

        # agregar bias
        bias = np.ones((X_val_np.shape[0], 1))
        X_val_np = np.hstack((bias, X_val_np))
        
        return X_val_np @ self.coef_inv
        
        
    def predict_grad(self, X_val):
        if self.coef_grad is None:
            raise ValueError("Modelo no entrenado con gradient descent")

        # convierto a np array
        X_val_np = X_val.values

        # agregar bias
        bias = np.ones((X_val_np.shape[0], 1))
        X_val_np = np.hstack((bias, X_val_np))
        
        return X_val_np @ self.coef_grad

    def show_coef_inv(self):

        for feature, w in zip(self.feature_names, self.coef_inv):
            print(f"{feature}: {w[0]:.2f}")

    def show_coef_grad(self):

        for feature, w in zip(self.feature_names, self.coef_grad):
            print(f"{feature}: {w[0]:.2f}")