# 🐧 Ubuntu 22 开发环境设置指南

## 前置需求

- Ubuntu 22.04 LTS
- sudo 权限

---

## 1️⃣ 系统依赖安装

首先更新系统包：

```bash
sudo apt update && sudo apt upgrade -y
```

### 安装必要的开发工具

```bash
sudo apt install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    curl \
    wget \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config
```

### 验证 Python 版本

```bash
python3 --version  # 应该是 Python 3.10+
pip3 --version
```

---

## 2️⃣ 创建虚拟环境

```bash
cd /home/lu/streamlit/DRFB

# 创建虚拟环境
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 升级 pip
pip install --upgrade pip setuptools wheel
```

---

## 3️⃣ 安装 Python 依赖

```bash
# 确保虚拟环境已激活
source .venv/bin/activate

# 安装基础依赖
pip install -r requirements.txt
pip install -r requirements_streamlit.txt
```

---

## 4️⃣ 安装 mbtrack2 (可选 - 需要 GitLab 权限)

如果你需要运行实际仿真（不仅仅是 UI 配置）：

```bash
# 克隆 mbtrack2
git clone https://gitlab.synchrotron-soleil.fr/pa/collective-effects/mbtrack2.git mbtrack2-stable

# 进入目录并安装
cd mbtrack2-stable
pip install -e .
cd ..
```

**如果无法访问 GitLab**，使用 Docker 版本（见下面第 5️⃣ 部分）。

---

## 5️⃣ 启动 Streamlit 应用

### 本地开发模式

```bash
# 确保虚拟环境已激活
source .venv/bin/activate

# 启动应用
streamlit run streamlit_app.py
```

应用将在 http://localhost:8501 打开。

### 使用 Docker 运行（推荐）

**安装 Docker：**

```bash
# Ubuntu 22 安装 Docker
sudo apt install -y docker.io docker-compose

# 将当前用户加入 docker 组（避免每次都用 sudo）
sudo usermod -aG docker $USER

# 重新登录或运行以下命令使组成员身份生效
newgrp docker
```

**构建并运行 Docker 镜像：**

```bash
cd /home/lu/streamlit/DRFB

# 方法 1: 使用 docker-compose（推荐）
docker-compose up --build

# 方法 2: 手动构建和运行
docker build -f Dockerfile.local -t albums-streamlit:latest .
docker run -p 8501:8501 albums-streamlit:latest
```

访问 http://localhost:8501

---

## 6️⃣ 开发工作流程

### 常用命令速查表

```bash
# 激活虚拟环境
source .venv/bin/activate

# 停用虚拟环境
deactivate

# 运行 Streamlit
streamlit run streamlit_app.py

# 运行测试
pytest tests/

# 检查代码风格
flake8 . --exclude=.venv,mbtrack2-stable,collective_effects

# 格式化代码
black . --exclude=.venv,mbtrack2-stable,collective_effects
```

### 使用 VS Code 开发

1. **安装 VS Code**
   ```bash
   sudo snap install code --classic
   ```

2. **安装推荐扩展**
   - Python (pylance)
   - Pylint
   - Black Formatter
   - Streamlit
   - Docker

3. **配置 Python 环境**
   - Ctrl+Shift+P → "Python: Select Interpreter"
   - 选择 `.venv/bin/python`

---

## 7️⃣ Docker 开发优势 (Ubuntu 22)

✅ **一致的环境** - 开发、测试和部署环境完全相同
✅ **快速部署** - 忘记手动配置依赖
✅ **隔离** - 不会影响系统其他应用
✅ **易于分享** - 其他人可以直接运行你的应用，无需安装

### Docker 工作流程

```bash
# 开发：修改代码后，自动 reload
docker-compose up

# 生产：构建完整镜像
./build_complete_docker.sh

# 分享镜像导出
docker save albums-streamlit:latest -o albums-streamlit.tar.gz
```

---

## 🔧 常见问题解决

### 问题 1: ImportError 关于 mbtrack2
**解决办法:**
- 如果无法安装 mbtrack2，使用 Docker 版本
- Docker 版本已包含所有依赖

### 问题 2: Streamlit 端口被占用
```bash
# 改用其他端口
streamlit run streamlit_app.py --server.port=8502
```

### 问题 3: Docker 权限错误
```bash
sudo usermod -aG docker $USER
newgrp docker
# 重新标签你的 shell
exec newgrp docker
```

### 问题 4: 内存不足
```bash
# 增加 Docker 可用内存（在 Docker Desktop 或 daemon.json 中）
# Ubuntu 上，Docker 使用总系统内存，调整如下：
sudo nano /etc/docker/daemon.json
```

---

## 📊 推荐的开发设置

| 工具 | 用途 | 安装方式 |
|------|------|--------|
| Python 3.10 | 运行时环境 | apt |
| venv | 虚拟环境 | 内置 |
| pip | 包管理 | 内置 |
| Docker | 容器化运行 | apt |
| VS Code | 代码编辑 | snap/官方网站 |
| Git | 版本控制 | apt |

---

## 🚀 快速开始（总结）

```bash
# 1. 进入项目目录
cd /home/lu/streamlit/DRFB

# 2. 激活虚拟环境
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
pip install -r requirements_streamlit.txt

# 4. 运行应用
streamlit run streamlit_app.py

# 或使用 Docker（推荐）
docker-compose up --build
```

---

## 💾 定期维护

```bash
# 更新所有包
pip list --outdated
pip install --upgrade -r requirements.txt

# 清理 Docker
docker system prune -a
```

---

## 📞 需要帮助？

查看其他相关指南：
- [DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md) - Docker 快速开始
- [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) - 完整安装说明
- [DOCKER_GUIDE.md](DOCKER_GUIDE.md) - Docker 详细指南
