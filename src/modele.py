import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


from src.donnees import COLONNES_ENTREE, CIBLE

GRAINE = 146
CHEMIN_MODELE = os.path.join("modeles", "modele.joblib")
CHEMIN_METADONNEES = os.path.join("modeles", "metadonnees.json")


def separer(df: pd.DataFrame, proportion_test: float = 0.2) -> tuple:
    X = df[COLONNES_ENTREE]
    y = df[CIBLE]
    return train_test_split(X, y, test_size=proportion_test, random_state=GRAINE)

def evaluer_reference(X_train, y_train, X_test, y_test) -> float:
    reference = DummyRegressor(strategy="median")
    reference.fit(X_train, y_train)
    predictions = reference.predict(X_test)
    return mean_absolute_error(y_test, predictions)




def entrainer(X_train, y_train) -> Pipeline:
    traduction = ColumnTransformer(
        [("marque", OneHotEncoder(handle_unknown="ignore"), ["Marque"])],
        remainder="passthrough",
    )
    modele = Pipeline([
        ("traduction", traduction),
        ("foret", RandomForestRegressor(n_estimators=100, random_state=GRAINE)),
    ])
    modele.fit(X_train, y_train)
    return modele


def evaluer(modele: Pipeline, X_test, y_test) -> dict:
    """Calcule la MAE et le R² du modèle sur les données de test."""
    predictions = modele.predict(X_test)
    return {
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }


def tracer_predit_contre_reel(modele: Pipeline, X_test, y_test):
    """Crée un graphique du prix prédit contre le prix réel.

    Renvoie la figure Matplotlib (on choisit ailleurs de l'enregistrer ou de l'afficher).
    """
    predictions = modele.predict(X_test)
    figure, axes = plt.subplots()
    axes.scatter(y_test, predictions, alpha=0.6, label="Appareils de test")
    maximum = max(y_test.max(), predictions.max())
    axes.plot([0, maximum], [0, maximum], color="red", label="Prédiction parfaite")
    axes.set_title("Prix prédit contre prix réel")
    axes.set_xlabel("Prix réel")
    axes.set_ylabel("Prix prédit")
    axes.legend()
    return figure

def creer_metadonnees(df: pd.DataFrame, scores: dict) -> dict:
    """Rassemble les infos dont l'interface a besoin pour se construire.

    Renvoie un dictionnaire : colonnes, liste des marques,
    minimum et maximum de chaque colonne numérique, scores du modèle.
    """
    intervalles = {}
    for colonne in COLONNES_ENTREE:
        if colonne != "Marque":
            intervalles[colonne] = {
                "min": float(df[colonne].min()),
                "max": float(df[colonne].max()),
            }
    return {
        "colonnes": COLONNES_ENTREE,
        "cible": CIBLE,
        "marques": sorted(df["Marque"].unique().tolist()),
        "intervalles": intervalles,
        "scores": scores,
    }


def sauvegarder(modele: Pipeline, metadonnees: dict) -> None:
    """Enregistre le modèle (joblib) et les métadonnées (JSON) dans modeles/."""
    os.makedirs("modeles", exist_ok=True)
    joblib.dump(modele, CHEMIN_MODELE)
    with open(CHEMIN_METADONNEES, "w", encoding="utf-8") as fichier:
        json.dump(metadonnees, fichier, indent=2, ensure_ascii=False)