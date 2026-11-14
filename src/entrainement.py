import os

from src.donnees import charger_donnees, nettoyer
from src.modele import (
    creer_metadonnees,
    entrainer,
    evaluer,
    evaluer_reference,
    sauvegarder,
    separer,
    tracer_predit_contre_reel,
)


def main() -> None:
    """Charge les données, entraîne le modèle, l'évalue et le sauvegarde."""
    df = nettoyer(charger_donnees())
    X_train, X_test, y_train, y_test = separer(df)

    scores = evaluer(entrainer(X_train, y_train), X_test, y_test)
    modele = entrainer(X_train, y_train)
    scores = evaluer(modele, X_test, y_test)
    scores["mae_reference"] = evaluer_reference(X_train, y_train, X_test, y_test)

    figure = tracer_predit_contre_reel(modele, X_test, y_test)
    figure.savefig(os.path.join("graphiques", "predit_contre_reel.png"))

    sauvegarder(modele, creer_metadonnees(df, scores))
    print("Modèle sauvegardé. Scores :", scores)


if __name__ == "__main__":
    main()