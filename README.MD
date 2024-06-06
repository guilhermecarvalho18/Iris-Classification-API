# Projeto de Classificação Iris

Este projeto fornece um modelo de aprendizado de máquina para classificar flores de íris usando o conjunto de dados Iris. O modelo é implantado como uma API REST usando FastAPI.

## Estrutura do Projeto
- `api/`: Contém a aplicação FastAPI.
- `data/`: Contém o conjunto de dados.
- `model/`: Contém o script de treinamento do modelo e o modelo serializado.
- `preprocessing/`: Contém o script de pré-processamento de dados.
- `Dockerfile`: Dockerfile para conteinerizar a aplicação.
- `requirements.txt`: Lista de dependências.

## Reproduzindo a pipeline

### 1. Configure o ambiente

1. **Clone o repositório**:
   ```
   git clone https://github.com/guilhermecarvalho18/Iris-Classification-API.git
   cd Iris-Classification-API
   ```

2. **Crie um ambiente virtual**:
   ```
   python3 -m venv venv
   source venv/bin/activate # No Windows, use `venv\Scripts\activate`
   ```

3. **Instalar dependências**:
   ```
   pip install -r requirements.txt
   ```

### 2. Prepare os dados

1. **Faça o download do dataset**:
   ```
   curl -o iris.csv https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
   mv iris.csv iris-classification-project/data/raw/
   ```

### 3. Treine o modelo

1. **Execute train.py**:
   ```
   python model/train.py
   ```

## Inicie a API

1. **Crie a Docker image:**

   ```
   docker build -t iris-classification-api .
   ```

2. **Execute o Docker container:**

   ```
   docker run -p 80:80 iris-classification-api
   ```
3. **API Endpoints**

   **Health Check**
   - Endpoint: /health
   - Método: GET
   - Descrição: Verifica o status de integridade da API.

   ```
   curl -X GET "http://localhost:80/health"
   ```

   **Predict**
    - Endpoint: /predict
    - Método: POST
    - Descrição : Prevê a classe de uma flor de íris.

   ```
   curl -X POST "http://localhost:80/predict" -H "Content-Type: application/json" -d @request.json
   ```
   - Estrutura de requisição para request.json:
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
   

