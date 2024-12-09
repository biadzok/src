from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score

def calculate_top_30_accuracy(y_true, combined_probs):
    """Calculate Top-30 Accuracy."""
    return top_k_accuracy_score(y_true, combined_probs, k=30)

def evaluate_model(y_true, y_pred):
    """Evaluate model performance."""
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
