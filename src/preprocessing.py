# Funciones para el preprocesamiento
import numpy as np

def preprocess(df):
    df = df.copy()

    df = handle_units(df)
    df = drop_zero_price(df)
    df = fill_pisos_nan(df)

    return df

def one_hot_encoder(train, val):

    known_types = train["tipo"].unique()

    train = train.copy()
    val = val.copy()

    for df in [train, val]:
        df["tipo_depto"] = 0
        df["tipo_ph"] = 0
        df["tipo_otro"] = 0

        df.loc[df["tipo"] == "depto", "tipo_depto"] = 1
        df.loc[df["tipo"] == "ph", "tipo_ph"] = 1

        mask_otro = ~df["tipo"].isin(known_types)
        df.loc[mask_otro, "tipo_otro"] = 1

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
            train_scaled[c] = 0
            val_scaled[c] = 0

        else:
            train_scaled[c] = (train[c] - means[c])/stds[c]
            val_scaled[c] = (val[c] - means[c])/stds[c]

    return train_scaled, val_scaled

def drop_zero_price(df):

    df = df[df["precio"].notna()]
    df = df[df["precio"] > 0]

    return df

def fill_edad_median(df_train, df_val):

    median = df_train["edad"].median()

    print("Edad filled with train median: ", median)

    df_train["edad"] = df_train["edad"].fillna(median)
    df_val["edad"] = df_val["edad"].fillna(median)

    return df_train, df_val

def fill_pisos_nan(df):

    mask = (df["tipo"] == "depto") & (df["pisos"].isna())
    df.loc[mask, "pisos"] = 0
    
    return df

def handle_units(df):

    # máscara para filas en ft2
    mask_ft2 = df["unidades"] == "ft2"

    # convertir a m2
    df.loc[mask_ft2, "Área"] = df.loc[mask_ft2, "Área"] * 0.092903
    df.loc[mask_ft2, "metros_cubiertos"] = df.loc[mask_ft2, "metros_cubiertos"] * 0.092903

    # eliminar columna unidades
    df = df.drop(columns=["unidades"])

    return df

