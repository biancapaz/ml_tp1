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

    def prepare_X_val(self, X_val):

        X_np = X_val.values
        bias = np.ones((X_np.shape[0], 1))
        return np.hstack((bias, X_np))

    def pseudo_inverse(self):
        #entrenamiento por pseudo-inversa

        xtx = self.X.T @ self.X
        ident = np.identity(self.X.shape[1])
        ident[0,0] = 0 # pongo en 0 para que no regularice bias

        pinv = np.linalg.pinv(xtx + self.l2 * ident)
        self.coef_inv = pinv @ self.X.T @ self.y
    
    def grad_mse(self, coef):
        # necesario para gradient_descent

        preds = self.X @ coef
        err = preds - self.y
        grad = (2 / self.X.shape[0]) * (self.X.T @ err)

        # Regularización L2 en gradiente (Ridge via GD)
        if self.l2 != 0:
            reg_l2 = 2 * self.l2 * coef.copy()
            reg_l2[0] = 0  # no regularizar el bias
            grad += reg_l2
        
        # Regularización L1 (Lasso via GD — no tiene solución analítica)
        if self.l1 != 0:
            reg_l1 = self.l1 * np.sign(coef)
            reg_l1[0] = 0 # no regularizo el bias
            grad += reg_l1

        return grad

    def gradient_descent(self, lr=0.001, epochs=1000, tol=1e-6):

        w = np.zeros((self.X.shape[1], 1))
        errors = []

        for epoch in range(epochs):
            grad = self.grad_mse(w)
            w = w - lr * grad

            y_pred = self.X @ w
            errors.append(mse(self.y, y_pred))

            # Early stopping: detener si el error ya no mejora significativamente
            if epoch > 0 and abs(errors[-1] - errors[-2]) < tol:
                break

        self.coef_grad = w
        return errors
    
    def predict(self, X_val):

        if self.coef_inv is not None:
            return self.predict_inv(X_val)
        elif self.coef_grad is not None:
            return self.predict_grad(X_val)
        else:
            raise ValueError("El modelo no ha sido entrenado. Llamar a pseudo_inverse() o gradient_descent() primero.")
    
    def predict_inv(self, X_val):

        if self.coef_inv is None:
            raise ValueError("Modelo no entrenado con pseudo-inversa")

        X_val_b = self.prepare_X_val(X_val)
        
        return X_val_b @ self.coef_inv
        
    def predict_grad(self, X_val):

        if self.coef_grad is None:
            raise ValueError("Modelo no entrenado con gradient descent")

        X_val_b = self.prepare_X_val(X_val)
        
        return X_val_b @ self.coef_grad

    def show_coef_inv(self):

        if self.coef_inv is None:
            print("Modelo no entrenado con pseudo-inversa.")
            return

        for feature, w in zip(self.feature_names, self.coef_inv):
            print(f"{feature}: {w[0]:.2f}")

    def show_coef_grad(self):

        if self.coef_grad is None:
            print("Modelo no entrenado con gradient descent.")
            return

        for feature, w in zip(self.feature_names, self.coef_grad):
            print(f"{feature}: {w[0]:.2f}")