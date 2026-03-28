# Funciones auxiliares

import numpy as np
from models import LinearRegression
from preprocessing import fill_edad_median, one_hot_encoder, normalize
from metrics import mse, rmse

def feature_engineering(df):
    df = df.copy()

    df["area_log"] = np.log(df["Área"] + 1)
    df["area_sq"] = df["Área"]**2
    df["densidad"] = df["metros_cubiertos"] / df["Área"]
    df["area_pileta"] = df["Área"] * df["pileta"]
    df["ambientes_por_area"] = df["ambientes"] / df["Área"]

    return df

def add_high_degree_features(df, cols, max_degree=8):

    df_new = df.copy()

    for c in cols:
        for d in range(2, max_degree+1):
            df_new[f"{c}^{d}"] = df_new[c] ** d

    return df_new

def learning_curve_model(train_df, val_df, model_name, best_l1=0, best_l2=0, lr=0.001, epochs=1000):

    sizes = np.linspace(0.1, 1.0, 10)
    train_errors = []
    val_errors = []

    for s in sizes:
        n = int(len(train_df) * s)
        subset = train_df.sample(n=n, random_state=42).copy()

        # target
        y_train = subset["precio"].values
        y_val = val_df["precio"].values
        X_train = subset.drop(columns=["precio"])
        X_val = val_df.drop(columns=["precio"])

        # select feature epending on model
        if model_name == "M1":
            X_train = X_train[["Área"]]
            X_val = X_val[["Área"]]

        # -------- M2 --------
        elif model_name == "M2":
            X_train = X_train[["Área", "pileta"]]
            X_val = X_val[["Área", "pileta"]]

        # -------- M3 --------
        elif model_name == "M3":
            X_train, X_val = fill_edad_median(X_train, X_val)
            X_train, X_val = one_hot_encoder(X_train, X_val)

            # 6 features
            features_M3 = ["Área", "metros_cubiertos" , "ambientes", "pileta", "lat", "edad"]
            X_train = X_train[features_M3]
            X_val = X_val[features_M3]

        # -------- M4 or M6 --------
        elif model_name == "M4" or model_name == "M6":
            X_train, X_val = fill_edad_median(X_train, X_val)
            X_train, X_val = one_hot_encoder(X_train, X_val)

            X_train = feature_engineering(X_train)
            X_val = feature_engineering(X_val)

            features_M4 = ["Área", "metros_cubiertos" ,"ambientes","pileta","lat","edad", "area_log", "area_sq", "densidad",
                "area_pileta", "ambientes_por_area"]
            
            X_train = X_train[features_M4]
            X_val = X_val[features_M4]

        # -------- M5 --------
        elif model_name == "M5":
            features_M5 = ["Área", "metros_cubiertos" ,"ambientes", "pileta", "lat", "edad"]

            X_train, X_val = fill_edad_median(X_train, X_val)
            X_train, X_val = one_hot_encoder(X_train, X_val)

            X_train = add_high_degree_features(X_train, cols=features_M5, max_degree=8)
            X_val = add_high_degree_features(X_val, cols=features_M5, max_degree=8)

        # normalize
        X_train, X_val = normalize(X_train, X_val)

        # train
        if model_name == "M6":
            # usar mejores lambdas
            model = LinearRegression(X_train, y_train, l2=best_l2, l1=best_l1)

            if best_l1 > 0:
                model.gradient_descent(lr=lr, epochs=epochs)
                y_pred_train = model.predict_grad(X_train)
                y_pred_val = model.predict_grad(X_val)
            else:
                model.pseudo_inverse()
                y_pred_train = model.predict_inv(X_train)
                y_pred_val = model.predict_inv(X_val)

        else:
            model = LinearRegression(X_train, y_train)

            model.pseudo_inverse()
            y_pred_train = model.predict_inv(X_train)
            y_pred_val = model.predict_inv(X_val)

        # errors
        train_errors.append(rmse(y_train, y_pred_train))
        val_errors.append(rmse(y_val, y_pred_val))

    return sizes, train_errors, val_errors