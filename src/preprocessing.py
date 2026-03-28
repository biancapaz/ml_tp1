# Funciones para el preprocesamiento

import numpy as np

def preprocess(df):
   
   # pipeline de preprocesameinto incial, se plica a cualquier set porque no hcae inferencias.
   df = df.copy()
   df = handle_units(df)
   df = drop_zero_price(df)
   df = fill_pisos_nan(df)
   return df

def finish_process(train, val):

    X_train_proc, X_val_proc = fill_edad_median(train, val)
    X_train_proc, X_val_proc = one_hot_encoder(X_train_proc, X_val_proc)
    X_train_norm, X_val_norm = normalize(X_train_proc, X_val_proc)

    return X_train_norm, X_val_norm

def one_hot_encoder(train, val):
    
    known_types = train["tipo"].unique()

    train = train.copy()
    val = val.copy()

    for df in [train, val]:
        df["tipo_depto"] = (df["tipo"] == "depto").astype(int)
        df["tipo_ph"]    = (df["tipo"] == "ph").astype(int)
        # tipo_otro = 1 si el tipo no fue visto en train
        df["tipo_otro"]  = (~df["tipo"].isin(known_types)).astype(int)
        df.drop(columns=["tipo"], inplace=True)

    return train, val

def normalize(train, val):
    # which columns will I normalize?
    columns = train.columns

    # calculate means and deviations
    means = {}
    stds = {}
    
    for c in columns:
        means[c] = train[c].mean()
        stds[c] = train[c].std()

    # generate sets copies
    train_scaled = train.copy()
    val_scaled = val.copy()

    # apply transformations

    for c in columns:
        if np.isclose(stds[c], 0):
            print(f"  [normalize] Advertencia: feature '{c}' tiene desvío ≈ 0 → se reemplaza por 0.")
            train_scaled[c] = 0.0
            val_scaled[c] = 0.0

        else:
            train_scaled[c] = (train[c] - means[c]) / stds[c]
            val_scaled[c] = (val[c] - means[c]) / stds[c]

    return train_scaled, val_scaled

def drop_zero_price(df):

    df = df[df["precio"].notna()]
    df = df[df["precio"] > 0]

    return df

def fill_edad_median(df_train, df_val):

    median = df_train["edad"].median()
    df_train = df_train.copy()
    df_val   = df_val.copy()
    df_train["edad"] = df_train["edad"].fillna(median)
    df_val["edad"] = df_val["edad"].fillna(median)

    return df_train, df_val

def fill_pisos_nan(df):

    mask = (df["tipo"] == "depto") & (df["pisos"].isna())
    df.loc[mask, "pisos"] = 0
    
    return df

def handle_units(df):

    known_units = df["unidades"].unique()
    for u in known_units:
        if u not in ("m2", "sqft"):
            print(f"  [handle_units] Unidad desconocida encontrada: '{u}' — no se convertirá.")
 
    mask_sqft = (df["unidades"] == "sqft")
    df.loc[mask_sqft, "Área"] = df.loc[mask_sqft, "Área"] * 0.092903
    df.loc[mask_sqft, "metros_cubiertos"] = df.loc[mask_sqft, "metros_cubiertos"] * 0.092903
 
    df = df.drop(columns=["unidades"])

    return df

"""ESTO FUE UN PROBLEMA

Horas antes de la entrega del trabajo detecte un error en la función de preprocesamiento `handle_units`, donde  intentaba identificar propiedades con unidades en squared feat utilizando el valor 'f2', cuando en realidad el dataset utilizaba 'sqft'. Como consecuencia, la conversión de superficie a metros cuadrados no 
se estaba realizando correctamente.

Este error generaba una inconsistencia importante en las variables de área (`Área` y `metros_cubiertos`), ya que coexistían observaciones en distintas unidades dentro del mismo dataset. Dado que estas variables son altamente relevantes para la predicción del precio, esta inconsistencia afectaba directamente el 
aprendizaje del modelo.

En particular, al no convertir correctamente las unidades, el modelo interpretaba superficies en distintas escalas como si fueran comparables, distorsionando las relaciones entre variables. Esto podía derivar en coeficientes poco interpretables o inconsistentes, así como en una aparente mejora o empeoramiento 
artificial del desempeño del modelo.

Una vez corregido el error y unificadas todas las superficies en metros cuadrados, los coeficientes del modelo cambiaron significativamente y pasaron a ser más coherentes ante la mezcla de los mercados de Buenos Aires y New York.

Quizas, de haberlo detectado antes habria llegado a hacer un mejor analisis de los resultados en los modelos.

"""

#def handle_units(df):
#
#    # máscara para filas en ft2
#    mask_ft2 = df["unidades"] == "ft2"
#
#    # convertir a m2
#    df.loc[mask_ft2, "Área"] = df.loc[mask_ft2, "Área"] * 0.092903
#    df.loc[mask_ft2, "metros_cubiertos"] = df.loc[mask_ft2, "metros_cubiertos"] * 0.092903
#
#    # eliminar columna unidades
#    df = df.drop(columns=["unidades"])
#
#    return df
