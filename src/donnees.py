import os
import pandas as pd
COLONNES_ENTREE = [
    "Marque", 
    "Release date", 
    "Max resolution",
    "Effective pixels", 
    "Zoom tele (T)", 
    "Weight (inc. batteries)", 
    "Storage included",
]

CIBLE = "Price"

def charger_donnees() -> pd.DataFrame:
    chemin = os.path.join("donnees", "camera_dataset.csv")
    if not os.path.exists(chemin):
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")
    with open(chemin, encoding="utf-8") as fichier:
        return pd.read_csv(fichier)

def nettoyer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Marque"] = df["Model"].str.split().str[0]
    df = df[COLONNES_ENTREE + [CIBLE]]
    df = df.dropna()
    df = df[df["Weight (inc. batteries)"] > 0]
    df = df[df["Max resolution"] > 0]
    return df

