import pandas as pd
import math
import numpy as np  # Import de numpy ajouté
from pathlib import Path
from data.load_data import load_observations, prepare_rf_data
from models.random_forest import train_random_forest, evaluate_random_forest
from models.cnn_model import prepare_cnn_data, create_cnn_model, train_cnn_model
from models.ensemble import combine_predictions_weighted
from evaluation.metrics import calculate_top_30_accuracy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = Path("C:/GEOLIFE_CLEF/data/raw")
IMAGE_PATH = Path("C:/GEOLIFE_CLEF/data/raw/patches-fr")

print("Chargement des données...")
train_obs, test_obs = load_observations(DATA_PATH)

print("Préparation des données Random Forest...")
train_data, test_data = prepare_rf_data(train_obs, test_obs)

# Réduction de la taille des données d'entraînement
sample_size = 50000  # Limitez à 50 000 échantillons
train_data = train_data.sample(n=sample_size, random_state=42)
X_train = train_data.drop(columns=["species_id", "observation_id", "subset"], errors="ignore")
y_train = train_data["species_id"]

X_test = test_data.drop(columns=["species_id", "observation_id", "subset"], errors="ignore")
y_test = test_data["species_id"]

# Convertir en numérique et remplacer les valeurs manquantes
X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

# Vérifiez les types des données
print("Types dans X_train :", X_train.dtypes)
print("Types dans X_test :", X_test.dtypes)

# Entraînement du Random Forest avec paramètres ajustés
rf_model = RandomForestClassifier(
    n_estimators=50,        # Réduisez le nombre d'arbres
    max_depth=15,           # Limitez la profondeur des arbres
    max_leaf_nodes=100,     # Limitez le nombre de feuilles par arbre
    min_samples_split=10,   # Nombre minimum d'échantillons pour diviser un nœud
    max_samples=0.5,        # Utilisez 50% des échantillons à chaque arbre
    random_state=42,
    n_jobs=-1               # Parallélisez sur tous les cœurs disponibles
)
print("Entraînement du Random Forest...")
rf_model.fit(X_train, y_train)
rf_probs = rf_model.predict_proba(X_test)

print("Préparation des données CNN...")
train_images, train_labels = prepare_cnn_data(train_obs, IMAGE_PATH)

# Préparation des images de test pour le CNN
# print("Préparation des images de test pour le CNN...")
# test_images, _ = prepare_cnn_data(test_obs, IMAGE_PATH)
# print(f"taille du set de test : {len(test_images)}")

# Séparation des données en ensembles d'entraînement et de validation
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

# Filtrage des labels inconnus dans l'ensemble de validation
unique_train_labels = set(train_labels)
filtered_val_images = []
filtered_val_labels = []

for img, label in zip(val_images, val_labels):
    if label in unique_train_labels:
        filtered_val_images.append(img)
        filtered_val_labels.append(label)

val_images = np.array(filtered_val_images)
val_labels = np.array(filtered_val_labels)
print(f"Ensemble de validation filtré : {len(val_labels)} exemples")

# Encodage des labels
encoder = LabelEncoder()
train_labels = encoder.fit_transform(train_labels)
val_labels = encoder.transform(val_labels)

# Création et entraînement du modèle CNN
num_classes = len(set(train_labels))
print(f"num_classes : {num_classes}")
cnn_model = create_cnn_model(num_classes=num_classes)
cnn_model = train_cnn_model(cnn_model, train_images, train_labels, val_images, val_labels)

# Prédictions CNN sur l'ensemble de test
# cnn_probs = cnn_model.predict(test_images)

print("Fusion des prédictions...")

# Ajustement des probabilités du Random Forest pour correspondre aux classes CNN
def adjust_probabilities(probs, target_classes):
    """Ajuste les probabilités pour correspondre aux classes cibles."""
    adjusted_probs = np.zeros((probs.shape[0], len(target_classes)))
    for i, cls in enumerate(target_classes):
        if cls in encoder.classes_:
            adjusted_probs[:, i] = probs[:, np.where(encoder.classes_ == cls)[0][0]]
    return adjusted_probs

target_classes = encoder.classes_  # Classes encodées pour le CNN
rf_probs_adjusted = adjust_probabilities(rf_probs, target_classes)

# Vérifiez que les dimensions correspondent
print("Dimensions après ajustement :", rf_probs_adjusted.shape, cnn_probs.shape)

# Fusion des prédictions
combined_probs = combine_predictions_weighted(rf_probs_adjusted, cnn_probs)
accuracy = calculate_top_30_accuracy(y_test, combined_probs)

print(f"Précision Top-30 : {accuracy}")
