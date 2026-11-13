import os
import pandas as pd
import matplotlib.pyplot as plt

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

df["Marque"] = df["Model"].str.split().str[0]

print("\nLes marques et leur nombre d'appareils :")
print(df["Marque"].value_counts())

print("\nNombre de zéros par colonne :")
print((df["Weight (inc. batteries)"] == 0).sum())
print((df["Dimensions"] == 0).sum())
print((df["Max resolution"] == 0).sum())
print((df["Storage included"] == 0).sum())

os.makedirs("graphiques", exist_ok=True)
plt.figure()
plt.hist(df["Price"], bins=50)
plt.title("Distribution des prix des appareils photo")
plt.xlabel("Prix")
plt.ylabel("Nombre d'appareils")
plt.savefig(os.path.join("graphiques", "distribution_prix.png"))
print("Graphique enregistré dans graphiques/distribution_prix.png")