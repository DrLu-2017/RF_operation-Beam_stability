# ALBuMS Full Mode Installation Guide

## Current Status

Your ALBuMS application is currently running in **UI Mode**:
- ✅ Configure parameters
- ✅ Save/Load configurations
- ✅ Use presets
- ❌ Cannot run physical simulations

To enable **Full Mode** (running actual simulations), you need to install the `mbtrack2` library.

---

## Installation Options

### Option 1: Install mbtrack2 from GitLab (Recommended)

`mbtrack2` is a particle tracking library developed at Synchrotron SOLEIL.

#### Steps:

1. **Check for access permissions**
   ```bash
   # mbtrack2 is typically hosted on SOLEIL's GitLab
   # You may need access permissions
   ```

2. **Clone the repository** (if you have permission)
   ```bash
   cd /home/lu/streamlit/DRFB
   git clone https://gitlab.synchrotron-soleil.fr/pa/collective-effects/mbtrack2.git mbtrack2-stable
   ```

3. **Install dependencies**
   ```bash
   source .venv/bin/activate
   cd mbtrack2-stable
   pip install -e .
   ```

---

### Option 2: Use Docker (Easiest, recommended for sharing)

If you primarily want to share the tool with others, **using Docker is highly recommended**:

#### Advantages:
- ✅ No need to manually install mbtrack2
- ✅ Completely consistent environment
- ✅ No configuration needed by the recipient
- ✅ Dockerfile is already configured

#### Steps:

1. **Install Docker** (if not already installed)
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install docker.io docker-compose
   
   # Add your user to the docker group (to avoid using sudo every time)
   sudo usermod -aG docker $USER
   # Log out and log back in for changes to take effect
   ```

2. **Use the project's Dockerfile**
   
   Your project already has a Dockerfile based on SOLEIL's mbtrack2 image:
   ```bash
   cd /home/lu/streamlit/DRFB
   docker build -t albums .
   docker run -p 8501:8501 albums
   ```

---

### Option 3: For Demonstration/Configuration Only (Current Mode)

If you **don't need to run actual simulations** and just want to:
- Configure parameters
- Save configurations
- Share configurations with others
- Instructional demonstrations

Then **the current UI Mode is sufficient**, and no mbtrack2 installation is required.

---

## Recommended Strategy

### If you are using it for yourself:
**→ Option 2 (Docker)** - Simplest and most reliable.

### If you are sharing it with others:
**→ Option 2 (Docker)** - Recipients only need to run `docker run`.

### If you only need to configure parameters:
**→ Option 3 (Current Mode)** - Ready to use, no extra installation as you already have it.

---

## Quick Decision Matrix

**Question 1: Do you need to run actual physical simulations?**
- Yes → Need to install mbtrack2 (Option 1 or 2)
- No → Current mode is sufficient (Option 3)

**Question 2: Do you have SOLEIL GitLab access?**
- Yes → You can use Option 1
- No → Use Option 2 (Docker)

**Question 3: Are you primarily sharing this with others?**
- Yes → Highly recommend Option 2 (Docker)
- No → Choose based on your needs

---

## Docker Quick Start (Recommended)

```bash
# 1. Install Docker
sudo apt update
sudo apt install docker.io

# 2. Build the image
cd /home/lu/streamlit/DRFB
sudo docker build -t albums .

# 3. Run the application
sudo docker run -p 8501:8501 albums

# 4. Access the application
# Open your browser and visit http://localhost:8501
```

---

## Need Help?

If you've decided which option to use, let me know, and I can help you:
1. Install Docker
2. Build the Docker image
3. Or configure another installation method
