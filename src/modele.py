import pandas as pd
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
