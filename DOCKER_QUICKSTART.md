# 🐳 Docker Complete Image Quickstart

## 🎯 Objective

Create a Docker image containing all dependencies, so others can use ALBuMS by simply running Docker, without having to install mbtrack2 and pycolleff.

---

## ⚡ Quick Start (3 Steps)

### Step 1: Build the Image

```bash
cd /home/lu/streamlit/DRFB
./build_complete_docker.sh
```

Follow the prompts:
- Select "1" (Use local dependencies)
- Select "y" (Export image)
- Select "y" (Compress image)

### Step 2: Test the Image

```bash
docker run -p 8501:8501 albums-streamlit:latest
```

Open your browser and visit http://localhost:8501

### Step 3: Share

Share the generated `albums-streamlit-complete-YYYYMMDD.tar.gz` file with others.

---

## 📦 What's in the Image

✅ **Complete Computing Environment**:
- Python 3.10
- mbtrack2 (v0.9.1) - Particle tracking library
- pycolleff (v0.3.0) - Collective effects library
- All Python dependencies

✅ **ALBuMS Application**:
- Streamlit UI
- All pages and features
- Preset configurations
- Example files

✅ **Ready to Use**:
- No need to install any dependencies
- No environment configuration required
- Start with a single command

---

## 📊 Expected Size

- **Docker Image**: ~2-3 GB
- **Compressed tar.gz**: ~800 MB - 1.2 GB

---

## 🚀 How Recipients Use It

### Method 1: Using the tar.gz file

```bash
# 1. Decompress
gunzip albums-streamlit-complete-20260201.tar.gz

# 2. Load the image
docker load -i albums-streamlit-complete-20260201.tar

# 3. Run
docker run -p 8501:8501 albums-streamlit:latest

# 4. Visit http://localhost:8501
```

### Method 2: Using Docker Hub (if uploaded)

```bash
# Run directly
docker run -p 8501:8501 drlu2017/albums-streamlit:latest
```

---

## 📋 Full Documentation

- **[DOCKER_GUIDE.md](DOCKER_GUIDE.md)** - Detailed Docker Guide
- **[README.md](README.md)** - Project Description
- **[INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)** - Local Installation Guide

---

## 🔍 Verify the Image

After building, verify that the image contains all dependencies:

```bash
docker run -it albums-streamlit:latest /bin/bash

# Test inside the container
python -c "import mbtrack2; print('✓ mbtrack2')"
python -c "from pycolleff.longitudinal_equilibrium import LongitudinalEquilibrium; print('✓ pycolleff')"
python -c "from albums.robinson import RobinsonModes; print('✓ ALBuMS')"

exit
```

---

## 💡 Tips

1. **First-time build** takes 10-20 minutes (depending on network speed).
2. **Subsequent builds** will be much faster (Docker cache).
3. **Compressing the image** can reduce file size by 50-70%.
4. **Using Docker Hub** is the most convenient way to share (no need to transfer large files).

---

## 🆘 Need Help?

Check [DOCKER_GUIDE.md](DOCKER_GUIDE.md) for detailed instructions and troubleshooting.
