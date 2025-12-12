import sys
from pathlib import Path
import numpy as np

# Path Fix
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.const import DatasetType
from src.config import DataConfig, SMDOptions, CICOptions
from src.data_loader.factory import DataLoaderFactory

def test_smd_loader(real_path):
    print(f"\n>>> Testing SMD Loader at: {real_path}")
    
    # NEW CONFIG STYLE
    config = DataConfig(
        dataset_type=DatasetType.SMD,
        data_path=str(real_path),
        window_size=10,
        smd=SMDOptions(entity_id="machine-1-1") # Isolated Options
    )
    
    try:
        loader = DataLoaderFactory.get_loader(config)
        (X_train, _), (X_test, y_test) = loader.get_data()
        
        # Validation
        print(f"✅ Loaded. X_train: {X_train.shape}")
        assert len(X_train.shape) == 3, "Must be 3D"
        assert X_train.shape[1] == 10, "Window dimension must be axis 1"
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()

def test_cic_loader(real_path):
    print(f"\n>>> Testing CIC Loader at: {real_path}")
    
    cic_cols = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets']
    
    # NEW CONFIG STYLE
    config = DataConfig(
        dataset_type=DatasetType.CIC,
        data_path=str(real_path),
        window_size=10,
        cic=CICOptions(selected_columns=cic_cols) # Isolated Options
    )
    
    try:
        loader = DataLoaderFactory.get_loader(config)
        (X_train, _), (X_test, y_test) = loader.get_data()
        print(f"✅ Loaded. X_train: {X_train.shape}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    # UPDATE PATHS HERE
    REAL_SMD_PATH = "data/SMD" 
    REAL_CIC_PATH = "data/CIC-DDoS2019/cicddos2019_dataset.csv" 

    test_smd_loader(REAL_SMD_PATH)
    test_cic_loader(REAL_CIC_PATH)