import os
import pandas as pd

chemin = os.path.join("donnees", "camera_dataset.csv")

df = pd.read_csv(chemin)

print("Taille (lignes, colonnes) :")
print(df.shape)

print("\nLes 5 premières lignes :")
print(df.head())

print("\nType de chaque colonne :")
print(df.dtypes)

print("\nValeurs manquantes par colonne :")
print(df.isna().sum())

print("\nStatistiques des colonnes numériques :")
print(df.describe())