# ML Model Deployment on GCP with FastAPI, Docker, and Kubeflow

This guide walks through the complete process of deploying machine learning models using FastAPI, Docker, and Kubeflow on Google Cloud Platform.

## 1. GCP VM Setup

### Create a VM Instance

```bash
# Create a VM instance with sufficient resources for Docker and Kubernetes
gcloud compute instances create ml-deployment-vm \
  --project=your-project-id \
  --zone=us-central1-a \
  --machine-type=e2-standard-4 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-ssd

# Add a label to the VM for organization
gcloud compute instances add-labels ml-deployment-vm \
  --project=your-project-id \
  --zone=us-central1-a \
  --labels=environment=development,project=ml-deployment

# Create a firewall rule to allow Minikube dashboard access (port 30000-32767 for NodePort services)
gcloud compute firewall-rules create allow-minikube-dashboard \
  --project=your-project-id \
  --direction=INGRESS \
  --priority=1000 \
  --network=default \
  --action=ALLOW \
  --rules=tcp:30000-32767 \
  --source-ranges=0.0.0.0/0 \
  --target-tags=ml-deployment-vm

# Add the network tag to the VM
gcloud compute instances add-tags ml-deployment-vm \
  --project=your-project-id \
  --zone=us-central1-a \
  --tags=ml-deployment-vm
```

### SSH into the VM

```bash
gcloud compute ssh ml-deployment-vm --project=your-project-id --zone=us-central1-a
```

## 2. Install Dependencies

### Install Docker

```bash
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Set up stable repository
echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Add current user to docker group to run without sudo
sudo usermod -aG docker $USER

# Verify Docker installation
docker --version

# Apply the group membership without logging out and in again
newgrp docker
```

### Install Python and FastAPI

```bash
# Install Python and pip
sudo apt-get install -y python3 python3-pip python3-venv

# Create a virtual environment
python3 -m venv ml_env
source ml_env/bin/activate

# Install FastAPI and dependencies
pip install "fastapi[standard]" pandas scikit-learn aiohttp
```

## 3. Convert Flask APIs to FastAPI

Let's create a project structure:

```bash
mkdir -p ml_deployment/{canary_model,main_model,elector}
cd ml_deployment
```

### Create FastAPI Version of Canary Model

Create `canary_model/app.py`:

```python
# FastAPI conversion of flask_canary.py
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from typing import Dict, List, Any

app = FastAPI(title="Canary Model API")

# Model training
iris = datasets.load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)
loaded_model = GaussianNB()
loaded_model.fit(X_train, Y_train)

class IrisData(BaseModel):
    s_l: float
    s_w: float
    p_l: float
    p_w: float

def predict_data(data_dict: Dict[str, float]) -> List[float]:
    data = pd.DataFrame(data_dict, index=[0])
    prediction = loaded_model.predict_proba(data)
    return prediction.tolist()[0]

@app.post("/predict")
async def predict(data: IrisData):
    try:
        prediction = {}
        data_dict = data.dict()
        prediction['Scores'] = predict_data(data_dict)
        prediction['Input'] = data_dict
        return prediction
    except Exception as ex:
        raise HTTPException(status_code=400, detail=str(ex))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
```

### Create FastAPI Version of Main Model

Create `main_model/app.py`:

```python
# FastAPI conversion of flask_json_api.py
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.model_selection import train_test_split

app = FastAPI(title="Main Model API")

# For this example, we'll train the model directly since we don't have the pickle file
iris = datasets.load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=1)
loaded_model = GaussianNB()
loaded_model.fit(X_train, Y_train)

class IrisData(BaseModel):
    s_l: float
    s_w: float
    p_l: float
    p_w: float

def predict_data(data_dict: Dict[str, float]) -> List[float]:
    data = pd.DataFrame(data_dict, index=[0])
    prediction = loaded_model.predict_proba(data)
    return prediction.tolist()[0]

@app.post("/predict")
async def predict(data: IrisData):
    try:
        prediction = {}
        data_dict = data.dict()
        prediction['Scores'] = predict_data(data_dict)
        prediction['Input'] = data_dict
        return prediction
    except Exception as ex:
        raise HTTPException(status_code=400, detail=str(ex))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
```

### Create FastAPI Version of Elector

Create `elector/app.py`:

```python
# FastAPI version of flask_elector.py
import asyncio
import json
import time
import aiohttp
import logging
import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

# Set up logging
FORMAT = "[%(asctime)-15s][%(levelname)-8s]%(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting Elector Service")

# Constants
DURACION_POR_LLAMADA = 5
DURACION_TOTAL = 10

app = FastAPI(title="Elector API")


class IrisData(BaseModel):
    s_l: float
    s_w: float
    p_l: float
    p_w: float


async def do_request(
    url: str, data: Dict[str, Any], timeout_peticion: float, session: aiohttp.ClientSession
):
    logger.info(f"[do_request][{url}] Calling web")
    try:
        async with session.post(url, timeout=timeout_peticion, json=data) as resp:
            respuesta = await resp.text()
            logger.info(f"[do_request][{url}] Returns result:{respuesta}")
            return [respuesta, url]  # esta respuesta se devulve cuando ya ha terminado la petición
    except asyncio.TimeoutError as ex:
        logger.warning(f"[do_request][{url}]Timeout captured:{ex}")
        return None
    except Exception as ex:
        logger.error(f"[do_request][{url}]Exception:{ex}")
        return None


async def esperar_respuestas(modelos):
    resultados = []
    tiempo_inicial = time.time()
    for completado in asyncio.as_completed(modelos):
        respuesta = await completado
        print(respuesta)
        resultados.append(respuesta)
        duracion = time.time() - tiempo_inicial
        if duracion > DURACION_TOTAL:
            logger.error("Se ha sobrepasado el tiempo de espera de respuestas")
            break
    return resultados


# Determine if we're running in Docker
def is_docker_env():
    # In Docker Compose, check if environment shows we're in Docker
    # This is a simple check - in Docker, the hostname is usually the container ID
    return os.path.exists("/.dockerenv")


# Get the appropriate URLs based on environment
def get_service_urls():
    if is_docker_env():
        # In Docker Compose, use the service names
        return {"canary": "http://canary:5001", "model": "http://model:5000"}
    else:
        # For local development
        return {"canary": "http://localhost:5001", "model": "http://localhost:5000"}


async def llamar_a_modelos(session, data):
    modelos_llamados = []

    # Get appropriate URLs based on environment
    urls = get_service_urls()

    logger.info(f"Running with URLs: canary={urls['canary']}, model={urls['model']}")

    modelos_llamados.append(
        do_request(f"{urls['canary']}/predict", data, DURACION_POR_LLAMADA, session)
    )
    modelos_llamados.append(
        do_request(f"{urls['model']}/predict", data, DURACION_POR_LLAMADA, session)
    )
    return modelos_llamados


def trata_resultados(resultados):
    """Elije qué llamada responde"""
    respuesta = "Sin resultado de modelos"
    urls = get_service_urls()
    model_url = f"{urls['model']}/predict"

    logger.info(f"Checking for response from {model_url}")

    # First, check if we have any responses at all
    valid_responses = [r for r in resultados if r is not None]
    if not valid_responses:
        logger.warning("No valid responses received from any model")
        return respuesta

    # Look for the main model response
    for resultado in resultados:
        if resultado is not None:
            logger.info(f"Processing result from {resultado[1]}")
            if resultado[1] == model_url:
                respuesta = resultado[0]
                logger.info("Using main model response")
                break

    # If we didn't get a response from the main model, use the first valid response
    if respuesta == "Sin resultado de modelos" and valid_responses:
        respuesta = valid_responses[0][0]
        logger.info(f"Falling back to response from {valid_responses[0][1]}")

    return respuesta


async def get_datos(data):
    resultados_tratados = []
    async with aiohttp.ClientSession() as session:
        logger.info("Llamando a los modelos")
        modelos_llamados = await llamar_a_modelos(session, data)

        logger.info("Esperando a los modelos")
        resultados = await esperar_respuestas(modelos_llamados)

        logger.info("Trata resultados")
        resultados_tratados = trata_resultados(resultados)

    return resultados_tratados


@app.post("/predict")
async def predict(data: IrisData):
    try:
        data_dict = data.dict()
        respuesta = await get_datos(data_dict)
        # Handle the case where we got no valid responses
        if respuesta == "Sin resultado de modelos":
            raise HTTPException(status_code=503, detail="No models available for prediction")
        return json.loads(respuesta)
    except json.JSONDecodeError as ex:
        logger.error(f"Error decoding JSON response: {str(ex)}")
        raise HTTPException(status_code=500, detail="Invalid response from models")
    except Exception as ex:
        logger.error(f"Error in predict endpoint: {str(ex)}")
        raise HTTPException(status_code=400, detail=str(ex))


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    logger.info("Elector starting")
    uvicorn.run(app, host="0.0.0.0", port=5002)
```

## 4. Set Up Dockerfiles

### Create requirements.txt

Create the file at project root:

```
fastapi[standard]==0.95.1
uvicorn
numpy==1.23.5
pandas==1.5.3
scikit-learn==1.2.2
aiohttp==3.8.4
httpx==0.24.0
```

### Create Dockerfile for Canary Model

Create `canary_model/Dockerfile`:

```dockerfile
FROM python:3.9-slim

RUN apt-get update && \
    apt-get install -y unoconv


WORKDIR /app

COPY ./canary_model/app.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5001
ENV PORT 5001

# Use python to execute the script instead of calling uvicorn directly
CMD exec uvicorn app:app --host 0.0.0.0 --port 5001
```

### Create Dockerfile for Main Model

Create `main_model/Dockerfile`:

```dockerfile
FROM python:3.9-slim

RUN apt-get update && \
    apt-get install -y unoconv


WORKDIR /app

COPY ./main_model/app.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
ENV PORT 5000

# Use python to execute the script instead of calling uvicorn directly
CMD exec uvicorn app:app --host 0.0.0.0 --port 5000
```

### Create Dockerfile for Elector

Create `elector/Dockerfile`:

```dockerfile
FROM python:3.9-slim

RUN apt-get update && \
    apt-get install -y unoconv


WORKDIR /app

COPY ./elector/app.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5002
ENV PORT 5002

# Use python to execute the script instead of calling uvicorn directly
CMD exec uvicorn app:app --host 0.0.0.0 --port 5002
```

## 5. Test Individual Services

```bash
# Test Canary Model
cd canary_model
uvicorn app:app --host 0.0.0.0 --port 5001 --reload

# In another terminal, test the API
curl -X POST "http://localhost:5001/predict" \
  -H "Content-Type: application/json" \
  -d '{"s_l":5.9,"s_w":3,"p_l":5.1,"p_w":1.8}'

# Test Main Model
cd ../main_model
uvicorn app:app --host 0.0.0.0 --port 5000 --reload

# Test the API
curl -X POST "http://localhost:5000/predict" \
  -H "Content-Type: application/json" \
  -d '{"s_l":5.9,"s_w":3,"p_l":5.1,"p_w":1.8}'

# Test Elector (requires the other services to be running)
cd ../elector
uvicorn app:app --host 0.0.0.0 --port 5002 --reload

# Test the API
curl -X POST "http://localhost:5002/predict" \
  -H "Content-Type: application/json" \
  -d '{"s_l":5.9,"s_w":3,"p_l":5.1,"p_w":1.8}'
```

## 6. Create Docker Compose and Test

Create `docker-compose.yml` at the project root:

```yaml
version: '3'
services:
  canary:
    build:
      context: .
      dockerfile: ./canary_model/Dockerfile
    ports:
      - "5001:5001"
    networks:
      - ml-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5001/docs"]
      interval: 30s
      timeout: 10s
      retries: 3

  model:
    build:
      context: .
      dockerfile: ./main_model/Dockerfile
    ports:
      - "5000:5000"
    networks:
      - ml-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/docs"]
      interval: 30s
      timeout: 10s
      retries: 3

  elector:
    build:
      context: .
      dockerfile: ./elector/Dockerfile
    ports:
      - "5002:5002"
    networks:
      - ml-network
    depends_on:
      - canary
      - model
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5002/docs"]
      interval: 30s
      timeout: 10s
      retries: 3

networks:
  ml-network:
    driver: bridge
```

### Test Docker Compose

```bash
# Build and start the services
docker-compose up --build

# Test in another terminal
curl -X POST "http://localhost:5002/predict" \
  -H "Content-Type: application/json" \
  -d '{"s_l":5.9,"s_w":3,"p_l":5.1,"p_w":1.8}'
```

## 7. Install Minikube

```bash
# Install minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Start minikube
minikube start --driver=docker --cpus=4 --memory=8g --disk-size=20g

# Verify installation
kubectl get nodes
minikube status
```

## 8. Deploy Models with Kubeflow

### Install Kubeflow

```bash
# Install kfctl
wget https://github.com/kubeflow/kfctl/releases/download/v1.2.0/kfctl_v1.2.0-0-gbc038f9_linux.tar.gz
tar -xvf kfctl_v1.2.0-0-gbc038f9_linux.tar.gz
sudo mv kfctl /usr/local/bin/

# Set up Kubeflow deployment
export KF_NAME=kubeflow
export BASE_DIR=${HOME}/kubeflow
export KF_DIR=${BASE_DIR}/${KF_NAME}
export CONFIG_URI="https://raw.githubusercontent.com/kubeflow/manifests/v1.2-branch/kfdef/kfctl_k8s_istio.v1.2.0.yaml"

# Create directories
mkdir -p ${KF_DIR}

# Download configuration file
cd ${KF_DIR}
kfctl apply -V -f ${CONFIG_URI}

# Check Kubeflow installation
kubectl get pods -n kubeflow
```

### Deploy Models to Kubeflow

First, let's create Kubernetes YAML files for our services.

Create `k8s/canary-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: canary
  namespace: kubeflow
spec:
  selector:
    matchLabels:
      run: canary
  replicas: 2
  template:
    metadata:
      labels:
        run: canary
    spec:
      containers:
      - name: canary
        image: ${YOUR_DOCKER_USERNAME}/canary:latest
        ports:
        - containerPort: 5001
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

Create `k8s/canary-svc.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: canary
  namespace: kubeflow
  labels:
    run: canary
spec:
  ports:
  - port: 5001
    protocol: TCP
  selector:
    run: canary
```

Create `k8s/model-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model
  namespace: kubeflow
spec:
  selector:
    matchLabels:
      run: model
  replicas: 2
  template:
    metadata:
      labels:
        run: model
    spec:
      containers:
      - name: model
        image: ${YOUR_DOCKER_USERNAME}/model:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

Create `k8s/model-svc.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: model
  namespace: kubeflow
  labels:
    run: model
spec:
  ports:
  - port: 5000
    protocol: TCP
  selector:
    run: model
```

Create `k8s/elector-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: elector
  namespace: kubeflow
spec:
  selector:
    matchLabels:
      run: elector
  replicas: 2
  template:
    metadata:
      labels:
        run: elector
    spec:
      containers:
      - name: elector
        image: ${YOUR_DOCKER_USERNAME}/elector:latest
        ports:
        - containerPort: 5002
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

Create `k8s/elector-svc.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: elector
  namespace: kubeflow
  labels:
    run: elector
spec:
  type: NodePort
  ports:
  - name: http
    port: 5002
    targetPort: 5002
    protocol: TCP
  selector:
    run: elector
```

### Build and Push Docker Images

```bash
# Log in to Docker Hub
docker login

# Build and push images
export DOCKER_USERNAME=your-dockerhub-username

# Canary model
docker build -t $DOCKER_USERNAME/canary:latest -f canary_model/Dockerfile .
docker push $DOCKER_USERNAME/canary:latest

# Main model
docker build -t $DOCKER_USERNAME/model:latest -f main_model/Dockerfile .
docker push $DOCKER_USERNAME/model:latest

# Elector
docker build -t $DOCKER_USERNAME/elector:latest -f elector/Dockerfile .
docker push $DOCKER_USERNAME/elector:latest
```

### Apply Kubernetes Configurations

```bash
# Update YAML files with your Docker username
sed -i "s/\${YOUR_DOCKER_USERNAME}/$DOCKER_USERNAME/g" k8s/*.yaml

# Apply the configurations
kubectl apply -f k8s/canary-deployment.yaml
kubectl apply -f k8s/canary-svc.yaml
kubectl apply -f k8s/model-deployment.yaml
kubectl apply -f k8s/model-svc.yaml
kubectl apply -f k8s/elector-deployment.yaml
kubectl apply -f k8s/elector-svc.yaml

# Check deployments
kubectl get deployments -n kubeflow
kubectl get services -n kubeflow
kubectl get pods -n kubeflow
```

### Access Services

```bash
# Get the NodePort for elector service
kubectl get svc elector -n kubeflow -o jsonpath='{.spec.ports[0].nodePort}'

# Access the elector service (replace NODE_PORT with the actual port)
NODE_PORT=$(kubectl get svc elector -n kubeflow -o jsonpath='{.spec.ports[0].nodePort}')
curl -X POST "http://$(minikube ip):$NODE_PORT/predict" \
  -H "Content-Type: application/json" \
  -d '{"s_l":5.9,"s_w":3,"p_l":5.1,"p_w":1.8}'
```

## 9. Create a KFServing Model Service (Optional Enhancement)

If you want to use Kubeflow's model serving capabilities:

```yaml
apiVersion: "serving.kubeflow.org/v1beta1"
kind: "InferenceService"
metadata:
  name: "iris-predictor"
  namespace: kubeflow
spec:
  predictor:
    tensorflow:
      storageUri: "gs://your-bucket/iris_model/"
```

## 10. Cleanup

When finished with your deployment:

```bash
# Stop and delete Minikube cluster
minikube stop
minikube delete

# Stop GCP instance when not in use
gcloud compute instances stop ml-deployment-vm --project=your-project-id --zone=us-central1-a

# Delete GCP instance when project is complete
gcloud compute instances delete ml-deployment-vm --project=your-project-id --zone=us-central1-a
```

## Additional Tips

1. Consider setting up CI/CD with Cloud Build or GitHub Actions to automate deployments
2. Use Helm charts for more complex Kubernetes deployments
3. Implement monitoring with Prometheus and Grafana
4. Set up logging with Fluentd, Elasticsearch, and Kibana
5. Implement A/B testing functionality in the elector service
6. Add authentication and TLS for production deployments
