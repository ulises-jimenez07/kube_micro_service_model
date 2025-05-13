# ML Model Deployment Project

This repository contains code and configuration for deploying machine learning models on Google Cloud Platform using FastAPI, Docker, and Kubeflow. It demonstrates a complete CI/CD pipeline for ML model deployment with canary releases.

## Project Overview

This project implements a three-component ML system:

1. **Main Model**: The primary ML model service (Iris classifier)
2. **Canary Model**: A secondary model service for A/B testing
3. **Elector Service**: A routing service that directs traffic between models

The architecture allows for safe deployments through canary releases, where a small percentage of traffic is routed to a new model version before full rollout.

The project uses Google Cloud Artifact Registry for container image storage and distribution, with Kubernetes configurations for deployment to GKE or Minikube.

## Repository Structure

```
├── canary_model/
│   ├── app.py             # FastAPI application for canary model
│   └── Dockerfile         # Docker configuration for canary model
├── main_model/
│   ├── app.py             # FastAPI application for main model
│   └── Dockerfile         # Docker configuration for main model
├── elector/
│   ├── app.py             # FastAPI application for elector service
│   └── Dockerfile         # Docker configuration for elector
├── k8s/                   # Kubernetes configuration files
│   ├── canary-deployment.yaml
│   ├── canary-svc.yaml
│   ├── model-deployment.yaml
│   ├── model-svc.yaml
│   ├── elector-deployment.yaml
│   └── elector-svc.yaml
├── docker-compose.yml     # Docker Compose configuration
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Getting Started

### Prerequisites

- Google Cloud Platform account
- Google Cloud SDK
- GCP Artifact Registry API enabled
- Docker and Docker Compose
- Python 3.9+

### Setting Up GCP VM

```bash
# Set up environment variables
export PROJECT_ID=your-gcp-project-id
export LOCATION=us-central1  # Choose appropriate region
export REPOSITORY=ml-models  # Name of your Artifact Registry repository

# Create a VM instance with sufficient resources for Docker and Kubernetes
gcloud compute instances create ml-deployment-vm \
  --project=$PROJECT_ID \
  --zone=$LOCATION-a \
  --machine-type=e2-standard-4 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-type=pd-ssd \
  --boot-disk-size=30GB \
  --scopes=https://www.googleapis.com/auth/cloud-platform

# Add a label to the VM for organization
gcloud compute instances add-labels ml-deployment-vm \
  --project=$PROJECT_ID \
  --zone=$LOCATION-a \
  --labels=environment=development,project=ml-deployment

# Create a firewall rule to allow Minikube dashboard access (port 30000-32767 for NodePort services)
gcloud compute firewall-rules create allow-minikube-dashboard \
  --project=$PROJECT_ID \
  --direction=INGRESS \
  --priority=1000 \
  --network=default \
  --action=ALLOW \
  --rules=tcp:30000-32767 \
  --source-ranges=0.0.0.0/0 \
  --target-tags=ml-deployment-vm

# Add the network tag to the VM
gcloud compute instances add-tags ml-deployment-vm \
  --project=$PROJECT_ID \
  --zone=$LOCATION-a \
  --tags=ml-deployment-vm

# SSH into the VM
gcloud compute ssh ml-deployment-vm --project=$PROJECT_ID --zone=$LOCATION-a
```

### Installing Dependencies

```bash
# Update package index
sudo apt-get update

# Install Docker prerequisites
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

# Apply the group membership without logging out and in again
newgrp docker

# Verify Docker installation
docker --version

# Install Docker compose
sudo apt-get  install docker-compose 

# Install Python and FastAPI
sudo apt-get install -y python3 python3-pip python3-venv

# Create a virtual environment
python3 -m venv ml_env
source ml_env/bin/activate

# Install FastAPI and dependencies
pip install "fastapi[standard]" pandas scikit-learn aiohttp
```

### Running Locally

1. Clone this repository and create project structure:
   ```bash
   git clone https://github.com/ulises-jimenez07/kube_micro_service_model.git
   cd kube_micro_service_model
   ```

2. Create the requirements.txt file:
   ```
    fastapi[standard]
    uvicorn
    numpy
    pandas
    scikit-learn
    aiohttp
    httpx
   ```

3. Create the FastAPI applications for each service (canary_model/app.py, main_model/app.py, elector/app.py) as shown in the guide.

4. Run individual services:
   ```bash
   # Run the canary model
   cd canary_model
   uvicorn app:app --host 0.0.0.0 --port 5001 --reload

   # In another terminal, run the main model
   cd ../main_model
   uvicorn app:app --host 0.0.0.0 --port 5000 --reload

   # In another terminal, run the elector
   cd ../elector
   uvicorn app:app --host 0.0.0.0 --port 5002 --reload
   ```

5. Test the API:
   ```bash
   curl -X POST "http://localhost:5001/predict" \
     -H "Content-Type: application/json" \
     -d '{"s_l":5.9,"s_w":3,"p_l":5.1,"p_w":1.8}'
   
   curl -X POST "http://localhost:5000/predict" \
     -H "Content-Type: application/json" \
     -d '{"s_l":5.9,"s_w":3,"p_l":5.1,"p_w":1.8}'
   
   curl -X POST "http://localhost:5002/predict" \
     -H "Content-Type: application/json" \
     -d '{"s_l":5.9,"s_w":3,"p_l":5.1,"p_w":1.8}'
   ```

### Running with Docker Compose

1. Create Dockerfiles for each service.

2. Create a docker-compose.yml file:
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

3. Build and start all services:
   ```bash
   docker-compose up --build
   ```

4. Test the API:
   ```bash
   curl -X POST "http://localhost:5002/predict" \
     -H "Content-Type: application/json" \
     -d '{"s_l":5.9,"s_w":3,"p_l":5.1,"p_w":1.8}'
   ```

## Kubernetes Deployment

### Setting Up Minikube

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

### Deploying to Kubernetes

1. Create Kubernetes YAML files in a k8s directory as described in the guide.

2. Build and push Docker images to GCP Artifact Registry:
   ```bash
   # Set up environment variables
   export PROJECT_ID=your-gcp-project-id
   export LOCATION=us-central1  # Choose appropriate region
   export REPOSITORY=ml-models  # Name of your Artifact Registry repository

   # Enable the Artifact Registry API
   gcloud services enable artifactregistry.googleapis.com

   # Create Artifact Registry repository if it doesn't exist
   gcloud artifacts repositories create $REPOSITORY \
     --repository-format=docker \
     --location=$LOCATION \
     --description="ML model container images"

   # Configure Docker to use Google Cloud as a credential helper
   gcloud auth configure-docker $LOCATION-docker.pkg.dev

   # Build and push images
   # Canary model
   docker build -t $LOCATION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/canary:latest -f canary_model/Dockerfile .
   docker push $LOCATION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/canary:latest

   # Main model
   docker build -t $LOCATION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/model:latest -f main_model/Dockerfile .
   docker push $LOCATION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/model:latest

   # Elector
   docker build -t $LOCATION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/elector:latest -f elector/Dockerfile .
   docker push $LOCATION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/elector:latest
   ```

3. Apply Kubernetes configurations:
   ```bash
   # Update YAML files with your Artifact Registry image paths
   sed -i "s|\${IMAGE_PLACEHOLDER}|$LOCATION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/canary:latest|g" k8s/canary-deployment.yaml
   sed -i "s|\${IMAGE_PLACEHOLDER}|$LOCATION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/model:latest|g" k8s/model-deployment.yaml
   sed -i "s|\${IMAGE_PLACEHOLDER}|$LOCATION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/elector:latest|g" k8s/elector-deployment.yaml

   # Create the kubeflow namespace
   kubectl create namespace kubeflow

   # Apply the configurations
   kubectl apply -f k8s/canary-deployment.yaml
   kubectl apply -f k8s/canary-svc.yaml
   kubectl apply -f k8s/model-deployment.yaml
   kubectl apply -f k8s/model-svc.yaml
   kubectl apply -f k8s/elector-deployment.yaml
   kubectl apply -f k8s/elector-svc.yaml

   # Check deployments (specify the kubeflow namespace)
   kubectl get deployments -n kubeflow
   kubectl get services -n kubeflow
   kubectl get pods -n kubeflow
   ```

4. Test the deployment:
   ```bash
   # Get the NodePort for elector service (specify the kubeflow namespace)
   NODE_PORT=$(kubectl get svc elector -n kubeflow -o jsonpath='{.spec.ports[0].nodePort}')
   
   # Access the elector service
   curl -X POST "http://$(minikube ip):$NODE_PORT/predict" \
     -H "Content-Type: application/json" \
     -d '{"s_l":5.9,"s_w":3,"p_l":5.1,"p_w":1.8}'
   ```

   If you encounter connection issues, try these troubleshooting steps:

   ```bash
   # Check if pods are running correctly
   kubectl get pods -n kubeflow
   
   # Check logs for any pod (replace pod-name with actual pod name)
   kubectl logs pod-name -n kubeflow
   
   # Alternative way to access the service using minikube
   minikube service elector -n kubeflow --url
   
   # Then use the URL provided by the above command
   curl -X POST "$(minikube service elector -n kubeflow --url)/predict" \
     -H "Content-Type: application/json" \
     -d '{"s_l":5.9,"s_w":3,"p_l":5.1,"p_w":1.8}'
   
   # If needed, ensure minikube tunnel is running (in a separate terminal)
   minikube tunnel
   ```

## Model Description

This project uses the Iris dataset to demonstrate ML model deployment:

1. **Main Model (GaussianNB)**: The primary model for Iris classification
2. **Canary Model (GaussianNB with different random state)**: A model version for comparison

Both models predict the species of an iris flower based on four measurements:

- `s_l`: Sepal Length (cm)
- `s_w`: Sepal Width (cm)
- `p_l`: Petal Length (cm)
- `p_w`: Petal Width (cm)

The models output probability distributions across the three Iris species.

The Elector service intelligently routes between these models with a preference for the main model.

## Troubleshooting Common Issues

### ImagePullBackOff Error

If you see `ImagePullBackOff` errors when checking your pods:

```bash
kubectl get pods -n kubeflow
```

This usually means Kubernetes can't pull the container images. This can happen if:

1. The `${IMAGE_PLACEHOLDER}` in the YAML files wasn't replaced with actual image paths
2. The specified images don't exist in the registry
3. Minikube doesn't have access to the registry

To fix this issue:

#### Option 1: Use local Docker images with Minikube

```bash
# Build the images locally
docker build -t canary:latest -f canary_model/Dockerfile .
docker build -t model:latest -f main_model/Dockerfile .
docker build -t elector:latest -f elector/Dockerfile .

# Load the images into Minikube
minikube image load canary:latest
minikube image load model:latest
minikube image load elector:latest

# Update the YAML files to use the local images
sed -i "s|\${IMAGE_PLACEHOLDER}|canary:latest|g" k8s/canary-deployment.yaml
sed -i "s|\${IMAGE_PLACEHOLDER}|model:latest|g" k8s/model-deployment.yaml
sed -i "s|\${IMAGE_PLACEHOLDER}|elector:latest|g" k8s/elector-deployment.yaml

# Apply the configurations again
kubectl apply -f k8s/canary-deployment.yaml
kubectl apply -f k8s/canary-svc.yaml
kubectl apply -f k8s/model-deployment.yaml
kubectl apply -f k8s/model-svc.yaml
kubectl apply -f k8s/elector-deployment.yaml
kubectl apply -f k8s/elector-svc.yaml
```

#### Option 2: Use Minikube's Docker daemon

```bash
# Configure your terminal to use Minikube's Docker daemon
eval $(minikube docker-env)

# Build the images directly in Minikube's Docker daemon
docker build -t canary:latest -f canary_model/Dockerfile .
docker build -t model:latest -f main_model/Dockerfile .
docker build -t elector:latest -f elector/Dockerfile .

# Update the YAML files to use the local images
sed -i "s|\${IMAGE_PLACEHOLDER}|canary:latest|g" k8s/canary-deployment.yaml
sed -i "s|\${IMAGE_PLACEHOLDER}|model:latest|g" k8s/model-deployment.yaml
sed -i "s|\${IMAGE_PLACEHOLDER}|elector:latest|g" k8s/elector-deployment.yaml

# Update the YAML files to never pull the images
# Add the following line under the image: line in each deployment YAML:
# imagePullPolicy: Never

# Apply the configurations again
kubectl apply -f k8s/canary-deployment.yaml
kubectl apply -f k8s/canary-svc.yaml
kubectl apply -f k8s/model-deployment.yaml
kubectl apply -f k8s/model-svc.yaml
kubectl apply -f k8s/elector-deployment.yaml
kubectl apply -f k8s/elector-svc.yaml
```

### Accessing GCP Artifact Registry from a Local Kubernetes Cluster

To access Google Cloud Platform (GCP) Artifact Registry from a local Kubernetes cluster using a service account key file, you need to follow these steps:

1. **Create a GCP Service Account and Key File**
2. **Create a Kubernetes Secret with the Service Account Key**
3. **Configure Your Kubernetes Deployment to Use the Secret**
4. **Pull Images from Artifact Registry**

#### Step-by-Step Guide

##### 1. Create a GCP Service Account and Key File

Create the Service Account:
```bash
gcloud iam service-accounts create my-service-account --display-name "My Service Account"
```

Grant the Necessary Roles to the Service Account:
```bash
gcloud projects add-iam-policy-binding <YOUR-PROJECT-ID> \
    --member="serviceAccount:my-service-account@<YOUR-PROJECT-ID>.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.reader"
```
Replace `<YOUR-PROJECT-ID>` with your GCP project ID.

Create and Download the Key File:
```bash
gcloud iam service-accounts keys create key.json \
    --iam-account my-service-account@<YOUR-PROJECT-ID>.iam.gserviceaccount.com
```

##### 2. Create a Kubernetes Secret with the Service Account Key

Create the Secret:
```bash
kubectl create secret docker-registry gcp-artifact-registry \
    --docker-server=LOCATION-docker.pkg.dev \
    --docker-username=_json_key \
    --docker-password="$(cat key.json)" \
    --docker-email=your-email@example.com
```
Replace:
- `LOCATION` with the location of your Artifact Registry (e.g., us-central1).
- `your-email@example.com` with your email.

##### 3. Configure Your Kubernetes Deployment to Use the Secret

Update your Kubernetes deployment YAML to reference the secret for pulling images.

Update Deployment YAML:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-app
        image: LOCATION-docker.pkg.dev/PROJECT-ID/REPOSITORY/IMAGE:TAG
        ports:
        - containerPort: 8080
      imagePullSecrets:
      - name: gcp-artifact-registry
```
Replace the placeholders:
- `LOCATION` with your Artifact Registry location (e.g., us-central1).
- `PROJECT-ID` with your GCP project ID.
- `REPOSITORY` with the name of your repository.
- `IMAGE:TAG` with the specific image and tag you want to use.

Apply the Deployment:
```bash
kubectl apply -f deployment.yaml
```

##### 4. Verify the Setup

Check the Deployment Status:
```bash
kubectl get pods
```

Describe a Pod to Verify Image Pull:
```bash
kubectl describe pod <POD-NAME>
```
Look for the events section to see if the image was pulled successfully.

#### Option 2: Use Local Images Instead (Recommended for Development)

The simplest solution is to use local images as described in the previous section, which avoids authentication issues entirely:

```bash
# Build the images locally
docker build -t canary:latest -f canary_model/Dockerfile .
docker build -t model:latest -f main_model/Dockerfile .
docker build -t elector:latest -f elector/Dockerfile .

# Update the YAML files to use the local images instead of GCP Artifact Registry
sed -i "s|us-central1-docker.pkg.dev/test-minikube-455501/ml-models/canary:latest|canary:latest|g" k8s/canary-deployment.yaml
sed -i "s|us-central1-docker.pkg.dev/test-minikube-455501/ml-models/model:latest|model:latest|g" k8s/model-deployment.yaml
sed -i "s|us-central1-docker.pkg.dev/test-minikube-455501/ml-models/elector:latest|elector:latest|g" k8s/elector-deployment.yaml

# Add imagePullPolicy: Never to each deployment
# Add this line right after the image: line in each deployment YAML

# Load the images into Minikube
minikube image load canary:latest
minikube image load model:latest
minikube image load elector:latest

# Apply the configurations again
kubectl apply -f k8s/canary-deployment.yaml
kubectl apply -f k8s/canary-svc.yaml
kubectl apply -f k8s/model-deployment.yaml
kubectl apply -f k8s/model-svc.yaml
kubectl apply -f k8s/elector-deployment.yaml
kubectl apply -f k8s/elector-svc.yaml
```

## Monitoring and Maintenance

For production deployments, consider implementing:

1. **Monitoring**: Set up Prometheus and Grafana
2. **Logging**: Configure EFK (Elasticsearch, Fluentd, Kibana) stack
3. **CI/CD**: Implement with GitHub Actions or Cloud Build
4. **A/B Testing**: Enhance the elector service for controlled experiments
5. **Security**: Add authentication and TLS for security

## Cleanup

When finished with your deployment:

```bash
# Stop and delete Minikube cluster
minikube stop
minikube delete

# Stop GCP instance when not in use
gcloud compute instances stop ml-deployment-vm --project=$PROJECT_ID --zone=$LOCATION-a

# Delete GCP instance when project is complete
gcloud compute instances delete ml-deployment-vm --project=$PROJECT_ID --zone=$LOCATION-a
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with FastAPI, Docker, and Kubernetes best practices
- Uses scikit-learn and the Iris dataset for ML model examples
- Follows MLOps principles for continuous deployment and monitoring
