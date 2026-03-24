# Docker Deployment Guide for HealthEye

## Prerequisites
- Docker and Docker Compose installed on your system
- Git (to clone the repository)

## Quick Start with Docker Compose

The easiest way to run HealthEye in Docker:

```bash
# Build and start the container
docker-compose up --build

# The app will be available at http://localhost:8501
```

## Building and Running with Docker

### Build the Docker image:
```bash
docker build -t healtheye:latest .
```

### Run the container:
```bash
docker run -p 8501:8501 healtheye:latest
```

### With volume mounts (for persisting models and datasets):
```bash
docker run -p 8501:8501 \
  -v $(pwd)/Models:/app/Models \
  -v $(pwd)/Dataset_models:/app/Dataset_models \
  healtheye:latest
```

## Important Notes

### CUDA/GPU Support
The current Dockerfile is configured with PyTorch CPU. For GPU support:

1. Use a CUDA-enabled base image:
   ```dockerfile
   FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
   ```

2. Or modify the requirements to pull the correct GPU version of PyTorch

### Environment Variables
You can set the following environment variables:
- `STREAMLIT_SERVER_PORT`: Port to run Streamlit on (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: 0.0.0.0)

```bash
docker run -p 8501:8501 \
  -e STREAMLIT_SERVER_PORT=8501 \
  healtheye:latest
```

### Docker Compose File
The `docker-compose.yml` includes:
- Automatic port mapping (8501:8501)
- Volume mounts for Models and Dataset_models
- Container restart policy
- Environment configuration

## Troubleshooting

**Container exits immediately:**
- Check logs: `docker logs <container-id>`
- Ensure all dependencies are properly installed in requirements.txt

**Port already in use:**
```bash
# Use a different port
docker run -p 8502:8501 healtheye:latest
```

**Memory issues:**
```bash
# Allocate more memory to Docker container
docker run -p 8501:8501 -m 4g healtheye:latest
```

## Production Deployment

For production, consider:
1. Using a `.env` file for secrets
2. Implementing a reverse proxy (nginx)
3. Adding health checks (already included in Dockerfile)
4. Using official Docker registries (Docker Hub, Amazon ECR, etc.)
5. Implementing CI/CD pipelines for automated builds
