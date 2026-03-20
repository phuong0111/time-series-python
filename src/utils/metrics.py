import numpy as np

def point_adjustment(y_true, y_pred):
    """
    Apply point adjustment to predictions.
    If a true anomaly segment has at least one detected point,
    the entire segment is considered detected.
    
    Args:
        y_true: Ground truth labels (1 for anomaly, 0 for normal)
        y_pred: Predicted labels (1 for anomaly, 0 for normal)
        
    Returns:
        y_pred_adjusted: Predictions after point adjustment
    """
    y_true = np.asarray(y_true)
    y_pred_adjusted = y_pred.copy()
    
    # Ensure boolean/integer arrays
    y_true_bool = (y_true == 1)
    
    # Find contiguous anomaly segments
    # Pad with 0s to easily find diffs
    padded_true = np.concatenate(([0], np.array(y_true_bool, dtype=int), [0]))
    diffs = np.diff(padded_true)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    
    # Apply PA for each segment
    for start, end in zip(starts, ends):
        # If there's any detection in this true anomaly segment
        if np.any(y_pred[start:end] == 1):
            y_pred_adjusted[start:end] = 1
            
    return y_pred_adjusted

def evaluate_with_pa(y_true, scores, steps=100):
    """
    Finds the optimal anomaly score threshold by maximizing Point-Adjusted F1 score.
    
    Args:
        y_true: True labels (samples,)
        scores: Anomaly scores (samples,)
        steps: Number of thresholds to evaluate
        
    Returns:
        Dictionary with best PA F1, Precision, Recall, and the threshold used.
    """
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    # We don't want to test exactly min or max, but spaced out between them
    # For speed, testing `steps` thresholds is usually enough
    step_size = (max_score - min_score) / steps
    thresholds = np.arange(min_score, max_score, step_size)
    
    best_pa_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_threshold = thresholds[0]
    
    # Only calculate if there are actually anomalies
    if np.sum(y_true) == 0:
        return {
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "threshold": thresholds[0]
        }
        
    for th in thresholds:
        # 1. Get binary predictions
        preds = (scores >= th).astype(int)
        
        # 2. Apply Point Adjustment
        pa_preds = point_adjustment(y_true, preds)
        
        # 3. Calculate metrics
        tp = np.sum((pa_preds == 1) & (y_true == 1))
        fp = np.sum((pa_preds == 1) & (y_true == 0))
        fn = np.sum((pa_preds == 0) & (y_true == 1))
        
        # Avoid division by zero
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        
        if f1 > best_pa_f1:
            best_pa_f1 = f1
            best_threshold = th
            best_precision = prec
            best_recall = rec
            
    return {
        "f1": best_pa_f1,
        "precision": best_precision,
        "recall": best_recall,
        "threshold": best_threshold
    }
