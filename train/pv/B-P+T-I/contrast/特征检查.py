import joblib
bundle = joblib.load("../../../../processed_data/model_ready_data_no_bp.pkl")
print("训练集特征维度:", bundle['train'][0].shape[1])
print("特征列名:", bundle.get('selected_features', '未保存'))
