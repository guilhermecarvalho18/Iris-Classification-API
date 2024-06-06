from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import sys
import os
import time
from typing import List, Union
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.data_preprocessor import DataPreprocessor

app = FastAPI()
startup_time = time.time()

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PreprocessingParams(BaseModel):
    missing_threshold: float = 0.7
    n_neighbors: int = 5
    variance_threshold: float = 0.0
    normalization_method: str = 'standard'
    apply_normalize: bool = True
    z_threshold: float = 3
    apply_filter_noise: bool = True
    pca_components: int = 2
    pca_variance_threshold: float = 0.95
    apply_pca: bool = True
    k_best_features: Union[int, str] = 'all'
    apply_select_features: bool = True
    encoding_method: str = 'label'
    drop_first: bool = True
    apply_encode_labels: bool = True

class ModelSelection(BaseModel):
    model_name: str
    preprocessing_params: PreprocessingParams
    features: List[IrisFeatures]

# Load all models
models = {
    'logistic_regression': joblib.load('model/saved_models/logistic_regression.pkl'),
    'svm': joblib.load('model/saved_models/svm.pkl'),
    'lda': joblib.load('model/saved_models/lda.pkl')
}

# Define class names
class_names = ['setosa', 'versicolor', 'virginica']

@app.post('/predict')
def predict(selection: ModelSelection):
    model_name = selection.model_name.lower()
    
    if model_name not in models:
        raise HTTPException(status_code=400, detail="Model not found. Available models: logistic_regression, svm, lda")
    
    model = models[model_name]
    
    features = selection.features
    params = selection.preprocessing_params
    
    data = [[
        f.sepal_length,
        f.sepal_width,
        f.petal_length,
        f.petal_width
    ] for f in features]
    df = pd.DataFrame(data, columns=['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth'])
    
    # Perform feature engineering based on user selection
    preprocessor = DataPreprocessor(filepath=None, missing_threshold=params.missing_threshold, n_neighbors=params.n_neighbors, variance_threshold=params.variance_threshold)
    preprocessor.data = df
    preprocessor.handle_missing_values()
    
    if params.apply_normalize:
        preprocessor.normalize_features(method=params.normalization_method)
    
    if params.apply_filter_noise:
        preprocessor.filter_noise(z_threshold=params.z_threshold)
    
    if params.apply_pca:
        preprocessor.apply_pca(n_components=params.pca_components, variance_threshold=params.pca_variance_threshold)
    
    if params.apply_select_features:
        preprocessor.select_features(k=params.k_best_features)
    
    if params.apply_encode_labels:
        preprocessor.encode_labels(method=params.encoding_method, drop_first=params.drop_first)
    
    predictions = model.predict(preprocessor.data)
    predictions_prob = model.predict_proba(preprocessor.data)
    
    results = []
    for i, prediction in enumerate(predictions):
        result = {
            'iris_index': i,
            'prediction': class_names[prediction],
            'scores': {class_name: score for class_name, score in zip(class_names, predictions_prob[i])}
        }
        results.append(result)
    
    return results

@app.get('/health')
def health_check():
    # Check model availability
    model_status = {model_name: os.path.isfile(f'model/saved_models/{model_name}.pkl') for model_name in models.keys()}
    
    # Uptime calculation
    current_time = time.time()
    uptime_seconds = current_time - startup_time
    uptime = {
        "days": int(uptime_seconds // (24 * 3600)),
        "hours": int((uptime_seconds % (24 * 3600)) // 3600),
        "minutes": int((uptime_seconds % 3600) // 60),
        "seconds": int(uptime_seconds % 60)
    }
    
    return {
        'status': 'ok',
        'models': model_status,
        'uptime': uptime,
        'api_version': '1.0.0'
    }
