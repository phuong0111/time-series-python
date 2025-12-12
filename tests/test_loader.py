import sys
from pathlib import Path
import numpy as np

# --- 1. SETUP PYTHON PATH ---
# Add the project root to sys.path so we can import 'src'
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.const import DatasetType
from src.config import DataConfig
from src.data_loader.factory import DataLoaderFactory

# --- 2. TEST CASES ---

def test_smd_loader(real_path):
    print(f"\n>>> Testing SMD Loader with Real Data at: {real_path}")
    
    # Check if path exists before running
    if not Path(real_path).exists():
        print(f"❌ Error: Path not found: {real_path}")
        return

    config = DataConfig(
        dataset_type=DatasetType.SMD,
        data_path=str(real_path),
        window_size=10,
        batch_size=32
    )
    
    try:
        loader = DataLoaderFactory.get_loader(config)
        (X_train, _), (X_test, y_test) = loader.get_data()
        
        print(f"✅ Success! Data Loaded.")
        print(f"   X_train shape: {X_train.shape}")
        print(f"   X_test shape:  {X_test.shape}")
        
        # Validation checks
        assert len(X_train.shape) == 3, "X_train must be 3D"
        assert X_train.shape[1] == 10, "Window size mismatch"
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()

def test_cic_loader(real_path):
    print(f"\n>>> Testing CIC Loader with Real Data at: {real_path}")
    
    if not Path(real_path).exists():
        print(f"❌ Error: File not found: {real_path}")
        return

    # Columns from your original notebook
    real_columns = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Fwd Packets Length Total', 'Bwd Packets Length Total',
        'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
        'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
        'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Max',
        'Fwd IAT Total', 'Fwd IAT Max', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count'
    ]
    
    config = DataConfig(
        dataset_type=DatasetType.CIC,
        data_path=str(real_path),
        selected_columns=real_columns,
        window_size=10,
        label_column='Class'
    )
    
    try:
        loader = DataLoaderFactory.get_loader(config)
        (X_train, _), (X_test, y_test) = loader.get_data()
        
        print(f"✅ Success! Data Loaded.")
        print(f"   X_train shape: {X_train.shape}") # Should be (N, 10, 20)
        print(f"   y_test shape:  {y_test.shape}")
        
        # Validation checks
        assert X_train.shape[2] == len(real_columns), f"Feature count mismatch. Expected {len(real_columns)}"
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    
    # ================= CONFIGURATION =================
    # UPDATE THESE PATHS TO MATCH YOUR REAL ENVIRONMENT
    # Example: If running on Colab, these might be /content/drive/MyDrive/...
    # Example: If local, these might be ./data/SMD
    
    REAL_SMD_PATH = "./data/SMD"  # Folder containing machine-1-1_train.csv
    REAL_CIC_PATH = "../CIC-DDoS2019/cicddos2019_dataset.csv" 
    # =================================================

    # Run tests
    test_smd_loader(REAL_SMD_PATH)
    # test_cic_loader(REAL_CIC_PATH)