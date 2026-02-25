# Docker Complete Image Build and Share Guide

This guide explains how to build a complete Docker image containing all dependencies (mbtrack2, pycolleff) for sharing with others.

## 📋 Prerequisites

Ensure the following directories exist in the project root:
- ✅ `mbtrack2-stable/` - mbtrack2 library
- ✅ `collective_effects/` - pycolleff library

These directories are already in your local environment and will be included in the Docker image.

---

## 🚀 Method 1: Using the Automation Script (Recommended)

### Step 1: Run the Build Script

```bash
cd /home/lu/streamlit/DRFB
./build_complete_docker.sh
```

The script will:
1. Check for dependency directories.
2. Build the Docker image.
3. Optional: Export the image to a tar file.
4. Optional: Compress the image file.

### Step 2: Test the Image

```bash
# Run the container
docker run -p 8501:8501 albums-streamlit:latest

# Access the application
# Open your browser and visit http://localhost:8501
```

### Step 3: Share the Image

If you chose to export the image, files similar to these will be generated:
- `albums-streamlit-complete-20260201.tar.gz` (Compressed)
- OR `albums-streamlit-complete-20260201.tar` (Uncompressed)

**Sharing with others**:
1. Send the tar.gz file to the recipient.
2. The recipient runs:
   ```bash
   # Decompress (if it's a .gz file)
   gunzip albums-streamlit-complete-20260201.tar.gz
   
   # Load the image
   docker load -i albums-streamlit-complete-20260201.tar
   
   # Run the application
   docker run -p 8501:8501 albums-streamlit:latest
   
   # Visit http://localhost:8501
   ```

---

## 🔧 Method 2: Manual Build

### Using Local Dependencies (Recommended)

```bash
# Build the image
docker build -f Dockerfile.local -t albums-streamlit:latest .

# Run
docker run -p 8501:8501 albums-streamlit:latest
```

### Downloading Dependencies from the Network

```bash
# Build the image (requires access to GitLab and GitHub)
docker build -f Dockerfile -t albums-streamlit:latest .

# Run
docker run -p 8501:8501 albums-streamlit:latest
```

---

## 📦 Exporting and Compressing the Image

### Export the Image

```bash
# Export to a tar file
docker save -o albums-streamlit.tar albums-streamlit:latest

# Check file size
du -h albums-streamlit.tar
```

### Compress the Image (Recommended for sharing)

```bash
# Compress the tar file
gzip albums-streamlit.tar

# This creates albums-streamlit.tar.gz
# Compressed size is typically reduced by 50-70%
```

---

## 🌐 Uploading to Docker Hub (Optional)

If you want to share via Docker Hub:

### Step 1: Login to Docker Hub

```bash
docker login
# Enter your Docker Hub username and password
```

### Step 2: Tag the Image

```bash
# Replace yourusername with your Docker Hub username
docker tag albums-streamlit:latest yourusername/albums-streamlit:latest
```

### Step 3: Push to Docker Hub

```bash
docker push yourusername/albums-streamlit:latest
```

### Step 4: For Others to Use

Others can directly run:
```bash
docker run -p 8501:8501 yourusername/albums-streamlit:latest
```

---

## 📊 Image Size Optimization

### The current image includes:
- ✅ Python 3.10
- ✅ Streamlit and all UI dependencies
- ✅ mbtrack2 (Particle tracking library)
- ✅ pycolleff (Collective effects library)
- ✅ All Python dependencies
- ✅ ALBuMS application code

### Expected Size:
- Uncompressed Image: ~2-3 GB
- Compressed: ~800 MB - 1.2 GB

### Suggestions to Reduce Image Size:
1. Use `.dockerignore` to exclude unnecessary files.
2. Use multi-stage builds (already implemented in the Dockerfile).
3. Clean up temporary files (already implemented in the Dockerfile).

---

## 🔍 Verify the Image

### Check if the image contains all dependencies

```bash
# Run the container and enter the shell
docker run -it albums-streamlit:latest /bin/bash

# Test inside the container
python -c "import mbtrack2; print('mbtrack2:', mbtrack2.__version__)"
python -c "from pycolleff.longitudinal_equilibrium import LongitudinalEquilibrium; print('pycolleff: OK')"
python -c "from albums.robinson import RobinsonModes; print('ALBuMS: OK')"

# Exit
exit
```

---

## 📝 Using docker-compose

Create `docker-compose.yml` (already provided):

```bash
# Start
docker-compose up

# Run in background
docker-compose up -d

# Stop
docker-compose down
```

---

## 🆘 Troubleshooting

### Issue 1: Build Failed - mbtrack2-stable not found

**Solution**:
```bash
# Ensure directories exist
ls -la mbtrack2-stable/
ls -la collective_effects/
```

### Issue 2: Image is too large

**Solution**:
- Use compression: `gzip albums-streamlit.tar`
- Or share via Docker Hub (no need to transfer files)

### Issue 3: Docker is not installed

**Solution**:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install docker.io docker-compose

# Add user to the docker group
sudo usermod -aG docker $USER
# Log out and log back in
```

---

## 📚 Recipient User Guide

If you are sharing with others, give them this simple guide:

### Using a tar file

```bash
# 1. Decompress (if it's a .gz file)
gunzip albums-streamlit-complete-YYYYMMDD.tar.gz

# 2. Load the image
docker load -i albums-streamlit-complete-YYYYMMDD.tar

# 3. Run the application
docker run -p 8501:8501 albums-streamlit:latest

# 4. Open browser to access
# http://localhost:8501
```

### Using Docker Hub

```bash
# Run directly (will download automatically)
docker run -p 8501:8501 yourusername/albums-streamlit:latest

# Visit http://localhost:8501
```

---

## ✅ Summary

**Recommended Sharing Workflow**:

1. **Build the image**:
   ```bash
   ./build_complete_docker.sh
   ```

2. **Choose sharing method**:
   - **File sharing**: Export and compress the tar file
   - **Docker Hub**: Push to Docker Hub

3. **Provide to recipient**:
   - tar.gz file + usage instructions
   - OR Docker Hub link

4. **Recipient uses**:
   - Load image or pull from Docker Hub
   - Run container
   - Access application

---

**Need help?** Check the project's GitHub Issues or contact the maintainers.
