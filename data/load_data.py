import pandas as pd
from pathlib import Path

def load_observations(data_path):
    """Load observation data for training and testing."""
    train_obs_path = data_path / "observations" / "observations_fr_train.csv"
    test_obs_path = data_path / "observations" / "observations_fr_test.csv"

    if not train_obs_path.exists():
        raise FileNotFoundError(f"Le fichier {train_obs_path} est introuvable.")
    if not test_obs_path.exists():
        raise FileNotFoundError(f"Le fichier {test_obs_path} est introuvable.")

    train_obs = pd.read_csv(train_obs_path, sep=";")
    test_obs = pd.read_csv(test_obs_path, sep=";")
    
    return train_obs, test_obs


def prepare_rf_data(train_obs, test_obs):
    """Prepare data for Random Forest."""
    env_vectors_path = Path("C:/GEOLIFE_CLEF/data/raw/pre-extracted/environmental_vectors.csv")
    landcover_path = Path("C:/GEOLIFE_CLEF/data/raw/metadata/landcover_suggested_alignment.csv")
    
    if not env_vectors_path.exists():
        raise FileNotFoundError(f"Le fichier {env_vectors_path} est introuvable.")
    if not landcover_path.exists():
        raise FileNotFoundError(f"Le fichier {landcover_path} est introuvable.")

    env_vectors = pd.read_csv(env_vectors_path, sep=";")
    landcover_data = pd.read_csv(landcover_path, sep=";")

    # Ajoutez un préfixe pour éviter les conflits
    env_vectors = env_vectors.add_prefix("env_")

    # Vérifiez les correspondances
    print("Nombre de correspondances dans train_obs :", train_obs['observation_id'].isin(env_vectors['env_observation_id']).sum())
    print("Nombre de correspondances dans test_obs :", test_obs['observation_id'].isin(env_vectors['env_observation_id']).sum())

    # Fusionnez les données environnementales avec observations
    train_data = pd.merge(train_obs, env_vectors, left_on="observation_id", right_on="env_observation_id", how="inner")
    test_data = pd.merge(test_obs, env_vectors, left_on="observation_id", right_on="env_observation_id", how="left")

    # Vérifiez si 'species_id' est manquant dans test_data
    if "species_id" not in test_data.columns:
        print("La colonne 'species_id' est absente dans test_data. Ajout d'une colonne vide.")
        test_data["species_id"] = None  # Ajout d'une colonne vide pour éviter les erreurs

    # Vérification des colonnes après fusion
    print("Colonnes dans train_data après fusion :", train_data.columns)
    print("Colonnes dans test_data après fusion :", test_data.columns)

    if "species_id" not in train_data.columns:
        raise ValueError("La colonne 'species_id' est manquante dans train_data.")

    return train_data, test_data



