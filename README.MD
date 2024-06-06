# Iris Classification Project

This project provides a machine learning model for classifying iris flowers using the Iris dataset. The model is deployed as a REST API using FastAPI.

## Project Structure
- `api/`: Contains the FastAPI application.
- `data/`: Contains the dataset.
- `model/`: Contains the model training script and serialized model.
- `preprocessing/`: Contains the data preprocessing script.
- `Dockerfile`: Dockerfile for containerizing the application.
- `requirements.txt`: List of dependencies.

## Reproducing the Pipeline

### 1. Set Up the Environment

1. **Clone the repository**:
   ```
   git clone https://github.com/guilhermecarvalho18/Iris-Classification-API.git
   cd Iris-Classification-API
   ```

2. **Create a virtual environment**:
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```


### 2. Prepare the Data

1. **Download the dataset**:
   ```
   curl -o iris.csv https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
   mv iris.csv iris-classification-project/data/raw/
   ```

### 3. Train the Model

1. **Run train.py**:
   ```
   python model/train.py
   ```

## Start the API

1. **Build the Docker image:**

   ```
   docker build -t iris-classification-api .
   ```

2. **Run the Docker container:**

   ```
   docker run -p 80:80 iris-classification-api
   ```
3. **API Endpoints**

   **Health Check**
   - Endpoint: /health
   - Method: GET
   - Description: Checks the health status of the API.

   ```
   curl -X GET "http://127.0.0.1:8000/health"
   ```

   **Predict**
    - Endpoint: /predict
    - Method: POST
    - Description: Predicts the class of an iris flower.

   ```
   curl -X POST "http://localhost:80/predict" -H "Content-Type: application/json" -d @request.json
   ```
   - Request Body for request.json:
   ```
    {
    "model_name": "logistic_regression",
    "preprocessing_params": {
        "missing_threshold": 0.7,
        "n_neighbors": 5,
        "variance_threshold": 0.0,
        "normalization_method": "standard",
        "apply_normalize": true,
        "z_threshold": 3,
        "apply_filter_noise": true,
        "pca_components": 2,
        "pca_variance_threshold": 0.95,
        "apply_pca": true,
        "k_best_features": "all",
        "apply_select_features": true,
        "encoding_method": "label",
        "drop_first": true,
        "apply_encode_labels": true
    },
    "features": [
        {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
        },
        {
        "sepal_length": 6.2,
        "sepal_width": 2.8,
        "petal_length": 4.8,
        "petal_width": 1.8
        }
    ]
    }
   ```
   

