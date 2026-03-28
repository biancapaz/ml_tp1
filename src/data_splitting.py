# Funciones para dividir los datos

import numpy as np
from models import LinearRegression
from preprocessing import fill_edad_median, one_hot_encoder, normalize
from utils import feature_engineering
from metrics import mse

def train_val_split(df, train_ratio=0.8, random_state=42):

    np.random.seed(random_state) # set a seed for reproducibility
    shuffled_indexes = np.random.permutation(len(df))

    # set train and val size

    train_size = int(len(df) * train_ratio)

    # set indexes
    train_idx = shuffled_indexes[:train_size]
    val_idx = shuffled_indexes[train_size:]

    # create subsets and reset indexes to start from 0 and drop old ones
    train_set = df.iloc[train_idx].copy().reset_index(drop=True)
    val_set = df.iloc[val_idx].copy().reset_index(drop=True)

    return train_set, val_set

def kfold_split(train, k=5, seed=42):
    np.random.seed(seed) # set a seed for reproducibility
    shuffled_indexes = np.random.permutation(len(train))

    folds = np.array_split(shuffled_indexes, k)

    return folds

def cross_val(df, lambdas, model_type, k=5, seed=42, lr=0.001, epochs=1000):

    features_M4 = [
        "Área", "metros_cubiertos", "ambientes", "pileta", "lat", "edad",
        "area_log", "area_sq", "densidad", "area_pileta", "ambientes_por_area"
    ]

    folds = kfold_split(df, k, seed)
    l_errors = {}

    for l in lambdas:

        fold_errors = []
        
        for i in range(k):

            # set indexes
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

            # create subsets and reset indexes to start from 0 and drop old ones
            val_set = df.iloc[val_idx].copy().reset_index(drop=True)
            train_set = df.iloc[train_idx].copy().reset_index(drop=True)

            # separate target
            y_train = train_set['precio'].values.copy()
            y_val = val_set['precio'].values.copy()
            X_train = train_set.drop(columns=["precio"])
            X_val = val_set.drop(columns=["precio"])

            # Pipeline de procesamiento (estadísticas SOLO de train del fold)
            # finishing processing data --> fill edad, one hot, feature engineering and normalization
            X_train, X_val = fill_edad_median(X_train, X_val)
            X_train, X_val = one_hot_encoder(X_train, X_val)
 
            X_train_fe = feature_engineering(X_train)
            X_val_fe   = feature_engineering(X_val)
 
            # Selección de las mismas features que en el modelo final
            X_train_fe = X_train_fe[features_M4]
            X_val_fe   = X_val_fe[features_M4]
 
            X_train_norm, X_val_norm = normalize(X_train_fe, X_val_fe)

            # train model depending on model_type
            if model_type == "ridge":
                model = LinearRegression(X_train_norm, y_train, l2=l, l1=0)
                model.pseudo_inverse()
                y_pred = model.predict_inv(X_val_norm)

            else: # model_type == "lasso"
                model = LinearRegression(X_train_norm, y_train, l2=0, l1=l)
                model.gradient_descent(lr=lr, epochs=epochs)
                y_pred = model.predict_grad(X_val_norm)

            # prediction with validation
            error = mse(y_val, y_pred)
            fold_errors.append(error)
        
        l_errors[l] = float(np.mean(fold_errors))
    
    best_l = min(l_errors, key=l_errors.get)

    return best_l, l_errors