# 📦 部署和分享指南

## 🎯 目标

本指南帮助你打包应用并与他人分享，确保在任何 Ubuntu 22 系统上都能无缝运行。

---

## 📋 预检清单

在部署前，确保以下条件满足：

- [ ] 所有代码已测试并提交到 Git
- [ ] 虚拟环境中所有依赖已安装
- [ ] `tests/` 目录中的测试全部通过
- [ ] 没有敏感信息（密钥、密码）在代码中
- [ ] `.gitignore` 已正确配置

---

## 方案 1: Docker 镜像分享（推荐）

### 步骤 1: 构建完整镜像

```bash
cd /home/lu/streamlit/DRFB

# 运行构建脚本
./build_ubuntu22_docker.sh

# 或使用 Make 命令
make docker-build
```

脚本会引导你完成以下步骤：
1. 验证依赖目录存在
2. 选择构建方法（推荐 Ubuntu 22）
3. 构建 Docker 镜像
4. 询问是否导出为文件

### 步骤 2: 导出镜像

```bash
# 导出为 tar 文件
docker save albums-streamlit-ubuntu22:latest -o albums-streamlit-ubuntu22.tar

# 压缩以减小文件大小（可选但推荐）
gzip albums-streamlit-ubuntu22.tar
# 结果文件: albums-streamlit-ubuntu22.tar.gz (~700 MB)
```

### 步骤 3: 分享镜像

- 上传到云存储（OneDrive、Google Drive、S3 等）
- 或通过文件传输服务分享
- 或创建 Docker Hub 账户并推送到公有/私有仓库

### 步骤 4: 接收方使用

接收方在任何装有 Docker 的机器上：

```bash
# 1. 解压（如果是 gzip）
gunzip albums-streamlit-ubuntu22.tar.gz

# 2. 加载镜像
docker load -i albums-streamlit-ubuntu22.tar

# 3. 运行
docker run -p 8501:8501 albums-streamlit-ubuntu22:latest

# 4. 浏览器打开
# http://localhost:8501
```

---

## 方案 2: GitHub 仓库分享

### 前置条件

- GitHub 账户
- Git 已安装并配置

### 步骤 1: 初始化 Git（如果还未初始化）

```bash
cd /home/lu/streamlit/DRFB
git init
git add .
git commit -m "Initial commit: DRFB Streamlit application"
```

### 步骤 2: 创建 GitHub 仓库

1. 登录 [GitHub](https://github.com)
2. 点击 "New" 创建新仓库
3. 命名（例如：`drfb-streamlit`）
4. 描述：DRFB Streamlit Application for RF System Analysis

### 步骤 3: 推送到 GitHub

```bash
# 添加远程 URL
git remote add origin https://github.com/YOUR_USERNAME/drfb-streamlit.git

# 推送代码
git branch -M main
git push -u origin main
```

### 步骤 4: 接收方克隆和运行

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/drfb-streamlit.git
cd drfb-streamlit

# 运行快速开始脚本
./start_ubuntu22.sh

# 或使用 Docker
make docker-dev
```

---

## 方案 3: 完整应用包分享

### 打包应用

```bash
cd /home/lu/streamlit/DRFB

# 创建压缩包（排除不必要文件）
tar --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    -czf drfb-streamlit-$(date +%Y%m%d).tar.gz .

# 验证包大小
ls -lh drfb-streamlit-*.tar.gz
```

### 内容清单

包中应包含：

```
✓ 源代码 (pages/, utils/, albums/, 等)
✓ 配置文件 (requirements.txt, Dockerfile, 等)
✓ 文档 (README.md, 设置指南, 等)
✓ 脚本 (start_ubuntu22.sh, Makefile, 等)
✓ 示例 (examples/ 目录)
✗ 虚拟环境 (.venv/) - 会自动创建
✗ 缓存文件 (__pycache__/) - 不需要
✗ 敏感数据 - 不包含
```

### 接收方解包和运行

```bash
# 解压
tar -xzf drfb-streamlit-YYYYMMDD.tar.gz
cd drfb-streamlit-YYYYMMDD

# 运行
./start_ubuntu22.sh
```

---

## 方案 4: Python 包分享（高级）

### 创建 setup.py

已在 `pyproject.toml` 中配置。可以转换为可安装的包：

```bash
# 安装构建工具
pip install build

# 构建包
python -m build

# 上传到 PyPI（如需发布）
pip install twine
twine upload dist/*
```

---

## 🔐 安全检查清单

分享前，确保没有泄露敏感信息：

```bash
# 1. 检查是否有密钥/密码
grep -r "password\|secret\|token\|key" . --exclude-dir=.git

# 2. 检查是否有敏感文件
grep -r "auth_config\|ssh_profiles" . --exclude-dir=.git

# 3. 检查 .gitignore
cat .gitignore

# 4. 验证虚拟环境不被跟踪
git status | grep venv
```

---

## 📊 分享方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **Docker 镜像** | 完全隔离、易部署、跨平台 | 文件较大（~700MB） | 生产部署、易分享 |
| **GitHub** | 版本控制、协作、免费托管 | 需要 Git 知识、依赖安装 | 团队开发、持续集成 |
| **应用包** | 文件较小、简单易用 | 环境依赖可能不同 | 一次性分发 |
| **Python 包** | 模块化、pip 安装 | 需要发布到 PyPI | 作为库供他人使用 |

---

## 🚀 自动化 CI/CD（可选）

### GitHub Actions 自动构建 Docker

创建 `.github/workflows/docker-build.yml`：

```yaml
name: Build Docker Image

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: docker build -f Dockerfile.ubuntu22 -t albums-streamlit-ubuntu22:latest .
      
      - name: Test image
        run: docker run --rm albums-streamlit-ubuntu22:latest --help
```

---

## 📈 部署到云平台

### Heroku 部署

```bash
# 1. 安装 Heroku CLI
curl https://cli.heroku.com/install.sh | sh

# 2. 登录
heroku login

# 3. 创建应用
heroku create my-drfb-app

# 4. 设置 Dockerfile
heroku stack:set container -a my-drfb-app

# 5. 部署
git push heroku main

# 访问应用
heroku open -a my-drfb-app
```

### AWS/Azure/Google Cloud

使用 Docker 镜像部署到：
- **AWS**: ECR + ECS / App Runner
- **Azure**: Container Registry + Container Instances
- **Google Cloud**: Artifact Registry + Cloud Run

---

## ✅ 验证清单

部署后验证：

- [ ] 应用在目标机器上启动成功
- [ ] Web UI 能正常访问 (http://localhost:8501)
- [ ] 所有页面都能加载
- [ ] 配置保存/加载功能正常
- [ ] 模型计算运行正确
- [ ] 没有错误日志输出

---

## 🐛 故障排查

### 镜像加载失败

```bash
# 检查磁盘空间
df -h

# 检查 Docker 版本
docker --version

# 清理旧镜像
docker system prune -a
```

### 应用启动缓慢

```bash
# 检查资源使用
docker stats

# 增加内存限制（在 docker-compose.yml）
mem_limit: 4g
```

### 性能问题

```bash
# 查看日志
docker logs -f <container_id>

# 限制缓存大小
docker exec <container_id> rm -rf /app/.streamlit/cache
```

---

## 📞 获取支持

如遇问题，参考：
- [UBUNTU_22_SETUP.md](UBUNTU_22_SETUP.md) - Ubuntu 22 设置
- [DOCKER_GUIDE.md](DOCKER_GUIDE.md) - Docker 详细说明
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - 快速参考

---

**上次更新**: 2026 年 3 月
