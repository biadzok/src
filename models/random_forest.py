from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, max_leaf_nodes=None, min_samples_split=2):
    """Train a Random Forest model."""
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_random_forest(model, X_test, y_test):
    """Evaluate Random Forest model."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    top_30_accuracy = top_k_accuracy_score(y_test, model.predict_proba(X_test), k=30)
    return accuracy, top_30_accuracy
