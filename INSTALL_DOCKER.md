# Docker Installation Guide

## 🐳 Installing Docker on Ubuntu/Debian

### Method 1: Using the Official Script (Easiest)

```bash
# Download and run the official Docker installation script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add the current user to the docker group (to avoid using sudo every time)
sudo usermod -aG docker $USER

# Log out and log back in, or run the following command for changes to take effect
newgrp docker

# Verify the installation
docker --version
docker run hello-world
```

### Method 2: Manual Installation (Recommended for production)

```bash
# 1. Update package index
sudo apt update

# 2. Install necessary packages
sudo apt install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# 3. Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# 4. Set up the Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 5. Update package index
sudo apt update

# 6. Install Docker Engine
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 7. Add user to the docker group
sudo usermod -aG docker $USER

# 8. Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# 9. Log out and log back in, then verify
docker --version
docker run hello-world
```

### Method 3: Using apt (Fastest)

```bash
# Install Docker
sudo apt update
sudo apt install -y docker.io docker-compose

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add user to the docker group
sudo usermod -aG docker $USER

# Log out and log back in, or run:
newgrp docker

# Verify
docker --version
```

---

## ✅ Verify Installation

After installation, run the following commands to verify:

```bash
# Check Docker version
docker --version

# Check Docker Compose version
docker-compose --version

# Run a test container
docker run hello-world

# Check Docker service status
sudo systemctl status docker
```

If you see "Hello from Docker!", the installation was successful!

---

## 🔧 Post-Installation Configuration

### Allow non-root users to run Docker

```bash
# Add current user to the docker group
sudo usermod -aG docker $USER

# Apply changes (choose one)
# Method 1: Log out and log back in
# Method 2: Run the following command
newgrp docker

# Verify (no sudo needed)
docker run hello-world
```

### Configure Docker to start on boot

```bash
sudo systemctl enable docker
```

---

## 🚀 After Installation

Once Docker is installed, return to the project directory and build the image:

```bash
cd /home/lu/streamlit/DRFB

# Build ALBuMS Docker image
./build_complete_docker.sh
```

---

## 🆘 Troubleshooting

### Issue 1: Permission Denied

**Error**: `permission denied while trying to connect to the Docker daemon socket`

**Solution**:
```bash
# Add user to the docker group
sudo usermod -aG docker $USER

# Log out and log back in, or run:
newgrp docker
```

### Issue 2: Docker Service Not Running

**Error**: `Cannot connect to the Docker daemon`

**Solution**:
```bash
# Start Docker service
sudo systemctl start docker

# Enable start on boot
sudo systemctl enable docker
```

### Issue 3: Port Already Allocated

**Error**: `port is already allocated`

**Solution**:
```bash
# Check which process is using the port
sudo lsof -i :8501

# Or use a different port
docker run -p 8502:8501 albums-streamlit:latest
```

---

## 📚 More Resources

- [Official Docker Documentation](https://docs.docker.com/engine/install/ubuntu/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Docker Hub](https://hub.docker.com/)

---

## 💡 Quick Command Reference

```bash
# List running containers
docker ps

# List all containers
docker ps -a

# Stop a container
docker stop <container_id>

# Remove a container
docker rm <container_id>

# List images
docker images

# Remove an image
docker rmi <image_id>

# View logs
docker logs <container_id>

# Enter a container
docker exec -it <container_id> /bin/bash
```

---

**After installation is complete, run `./build_complete_docker.sh` to begin building your ALBuMS image!**
