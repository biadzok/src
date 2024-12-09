import numpy as np

def combine_predictions_weighted(rf_probs, cnn_probs, rf_weight=0.3, cnn_weight=0.7):
    """Combine predictions from RF and CNN with weights."""
    return (rf_probs * rf_weight) + (cnn_probs * cnn_weight)
