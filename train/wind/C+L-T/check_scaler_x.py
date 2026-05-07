from joblib import load
import numpy as np

scaler_x = load("assets/wind_ceemdan_lgbm_trans/scaler_x")
print(f"scaler_x 期望特征数: {scaler_x.n_features_in_}")

selected_features = np.load("assets/wind_ceemdan_lgbm_trans/selected_features_indices.npy")
print(f"选择后的特征数: {len(selected_features)}")