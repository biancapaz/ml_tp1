# Funciones auxiliares

#save_results()     Para guardar resultados
#load_model()       Para cargar un modelo guardado

import numpy as np
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