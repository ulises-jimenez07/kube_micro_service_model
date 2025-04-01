# ML Model Deployment Project

This repository contains code and configuration for deploying machine learning models on Google Cloud Platform using FastAPI, Docker, and Kubeflow. It demonstrates a complete CI/CD pipeline for ML model deployment with canary releases.

## Project Overview

This project implements a three-component ML system:

1. **Main Model**: The primary ML model service (Iris classifier)
2. **Canary Model**: A secondary model service for A/B testing  
3. **Elector Service**: A routing service that directs traffic between models

The architecture allows for safe deployments through canary releases, where a small percentage of traffic is routed to a new model version before full rollout.

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
- Docker and Docker Compose
- Python 3.9+

### Setting Up GCP VM

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

# SSH into the VM
gcloud compute ssh ml-deployment-vm --project=your-project-id --zone=us-central1-a
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
   mkdir -p ml_deployment/{canary_model,main_model,elector}
   cd ml_deployment
   ```

2. Create the requirements.txt file:
   ```
   fastapi[standard]==0.95.1
   pandas==1.5.3
   scikit-learn==1.2.2
   aiohttp==3.8.4
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

1. Create Dockerfiles for each service as shown in the guide.

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

2. Build and push Docker images:
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

3. Apply Kubernetes configurations:
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
   kubectl get deployments
   kubectl get services
   kubectl get pods
   ```

4. Test the deployment:
   ```bash
   # Get the NodePort for elector service
   NODE_PORT=$(kubectl get svc elector -o jsonpath='{.spec.ports[0].nodePort}')
   
   # Access the elector service
   curl -X POST "http://$(minikube ip):$NODE_PORT/predict" \
     -H "Content-Type: application/json" \
     -d '{"s_l":5.9,"s_w":3,"p_l":5.1,"p_w":1.8}'
   ```

## Kubeflow Integration

For machine learning operations (MLOps), this project integrates with Kubeflow.

### Installing Kubeflow

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

# Download configuration file and apply
cd ${KF_DIR}
kfctl apply -V -f ${CONFIG_URI}

# Check Kubeflow installation
kubectl get pods -n kubeflow
```

### Deploying Models with Kubeflow

The Kubernetes YAML files in the guide already include the `namespace: kubeflow` specification for deployment to the Kubeflow namespace.

Optional: For more advanced Kubeflow integration, you can use KFServing:

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

## API Documentation

All services expose API documentation at the `/docs` endpoint using FastAPI's Swagger UI:

- Main Model: `http://localhost:5000/docs`
- Canary Model: `http://localhost:5001/docs`
- Elector Service: `http://localhost:5002/docs`

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
gcloud compute instances stop ml-deployment-vm --project=your-project-id --zone=us-central1-a

# Delete GCP instance when project is complete
gcloud compute instances delete ml-deployment-vm --project=your-project-id --zone=us-central1-a
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with FastAPI, Docker, and Kubernetes best practices
- Uses scikit-learn and the Iris dataset for ML model examples
- Follows MLOps principles for continuous deployment and monitoring
