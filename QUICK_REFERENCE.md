# 🐧 Ubuntu 22 快速参考卡片

## ⚡ 最快开始方式（3 步）

```bash
# 1. 进入项目
cd /home/lu/streamlit/DRFB

# 2. 运行快速开始脚本
./start_ubuntu22.sh

# 3. 选择运行方式（按提示）
```

---

## 🚀 常用命令

### 本地开发（虚拟环境）

```bash
# 一键设置
make setup

# 启动开发服务器
make dev

# 运行测试
make test

# 代码检查
make lint

# 代码格式化
make format
```

### Docker 开发（实时编辑）

```bash
# 启动开发模式（支持热重载）
make docker-dev

# 访问地址: http://localhost:8502
```

### Docker 生产（完全隔离）

```bash
# 启动生产模式
make docker-prod

# 访问地址: http://localhost:8501
```

### Docker 镜像管理

```bash
# 构建 Ubuntu 22 镜像
make docker-build

# 导出到文件（用于分享）
./build_ubuntu22_docker.sh
```

---

## 📋 完整命令列表

| 命令 | 说明 |
|------|------|
| `make help` | 显示所有可用命令 |
| `make setup` | 完整环境设置 |
| `make install` | 仅安装依赖 |
| `make dev` | 本地开发模式 |
| `make docker-dev` | Docker 开发模式 |
| `make docker-prod` | Docker 生产模式 |
| `make docker-build` | 构建 Docker 镜像 |
| `make test` | 运行测试 |
| `make lint` | 代码风格检查 |
| `make format` | 自动格式化代码 |
| `make clean` | 清理临时文件 |
| `make docker-clean` | 清理 Docker 资源 |

---

## 🔧 虚拟环境日常使用

### 激活虚拟环境

```bash
source .venv/bin/activate
```

### 停用虚拟环境

```bash
deactivate
```

### 查看已安装包

```bash
pip list
```

### 更新所有包

```bash
pip list --outdated
pip install --upgrade -r requirements.txt
```

---

## 🐳 Docker 常用操作

### 查看运行中的容器

```bash
docker ps
```

### 进入容器（调试）

```bash
docker exec -it albums-streamlit-dev bash
```

### 查看容器日志

```bash
docker logs -f albums-streamlit-dev
```

### 停止容器

```bash
docker-compose down
```

### 删除镜像

```bash
docker rmi albums-streamlit-ubuntu22:latest
```

---

## 🌐 访问地址

| 方式 | 地址 |
|------|------|
| 本地开发 | http://localhost:8501 |
| Docker 开发 | http://localhost:8502 |
| Docker 生产 | http://localhost:8501 |

---

## 💾 项目文件结构

```
/home/lu/streamlit/DRFB/
├── streamlit_app.py          # 主应用入口
├── pages/                     # 多页面应用
├── utils/                     # 工具函数
├── albums/                    # 核心库
├── examples/                  # 示例 notebook
│
├── requirements.txt           # 主要依赖
├── requirements_streamlit.txt # Streamlit 依赖
│
├── Dockerfile                 # 网络下载版本
├── Dockerfile.local           # 本地依赖版本
├── Dockerfile.ubuntu22        # Ubuntu 22 优化版
│
├── docker-compose.yml         # 生产配置
├── docker-compose.dev.yml     # 开发配置
│
├── Makefile                   # 快速命令
├── start_ubuntu22.sh          # 快速开始脚本
├── build_ubuntu22_docker.sh   # Docker 构建脚本
│
├── .devcontainer/             # VS Code 容器配置
├── UBUNTU_22_SETUP.md         # 详细设置指南
└── ...
```

---

## ❓ 常见问题快速解答

### Q: 如何快速启动应用？
A: 运行 `./start_ubuntu22.sh`

### Q: 本地开发和 Docker 的区别？
A: 
- **本地开发**: 快速迭代，需手动管理依赖
- **Docker**: 完全隔离，依赖一致，更易分享

### Q: 端口被占用怎么办？
A: 编辑 `docker-compose.dev.yml` 或 `.env`，改为其他端口

### Q: 如何分享应用给他人？
A: 
```bash
./build_ubuntu22_docker.sh  # 构建并导出镜像
# 分享 albums-streamlit-ubuntu22-YYYYMMDD.tar.gz
```

### Q: mbtrack2 安装失败？
A: 使用 Docker 版本，已包含所有依赖
```bash
make docker-dev
```

---

## 📚 详细文档

- [UBUNTU_22_SETUP.md](UBUNTU_22_SETUP.md) - Ubuntu 22 完整设置指南
- [DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md) - Docker 快速指南
- [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) - 完整安装说明
- [DOCKER_GUIDE.md](DOCKER_GUIDE.md) - Docker 详细指南

---

## 🎯 推荐工作流

### 开发阶段
```bash
# 方案 1: 本地虚拟环境（快速迭代）
make setup
make dev

# 方案 2: Docker（完全隔离）
make docker-dev
```

### 测试/部署
```bash
make test
make docker-build
```

### 分享给他人
```bash
./build_ubuntu22_docker.sh
# 选择导出和压缩选项
```

---

## 📞 获取帮助

查看帮助信息：
```bash
make help
```

查看详细指南：
```bash
cat UBUNTU_22_SETUP.md
```

---

**上次更新**: 2026 年 3 月  
**推荐系统**: Ubuntu 22.04 LTS  
**Python 版本**: 3.10+  
**Docker**: 20.10+
