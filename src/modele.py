import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error


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
