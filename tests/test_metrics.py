import numpy as np
import sys
from pathlib import Path

# Path setup
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.metrics import point_adjustment, evaluate_with_pa

def test_point_adjustment():
    # Ground truth: 1 anomaly segment [1:4], 1 point anomaly [6]
    y_true = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    
    # Prediction: catches only index 2, misses 6
    y_pred = np.array([0, 0, 1, 0, 0, 0, 0, 0])
    
    adjusted = point_adjustment(y_true, y_pred)
    
    # Expected: The entire segment [1, 1, 1] goes to 1, the 6 stays 0
    expected = np.array([0, 1, 1, 1, 0, 0, 0, 0])
    
    np.testing.assert_array_equal(adjusted, expected)
    print("test_point_adjustment passed")

def test_evaluate_with_pa():
    y_true = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    scores = np.array([0.1, 0.2, 0.9, 0.3, 0.1, 0.1, 0.4, 0.1])
    
    # If threshold is 0.8:
    # y_pred will be [0, 0, 1, 0, 0, 0, 0, 0]
    # after PA: [0, 1, 1, 1, 0, 0, 0, 0]
    # TP = 3, FP = 0, FN = 1
    # Precision = 1.0, Recall = 0.75, F1 = ~0.857
    
    res = evaluate_with_pa(y_true, scores, steps=10)
    print("evaluate_with_pa results:", res)
    assert res['threshold'] > 0.0
    print("test_evaluate_with_pa passed")

if __name__ == "__main__":
    test_point_adjustment()
    test_evaluate_with_pa()
