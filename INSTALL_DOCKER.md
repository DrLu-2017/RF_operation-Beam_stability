# Docker 安装指南

## 🐳 在 Ubuntu/Debian 上安装 Docker

### 方法 1：使用官方脚本（最简单）

```bash
# 下载并运行 Docker 官方安装脚本
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 将当前用户添加到 docker 组（避免每次都用 sudo）
sudo usermod -aG docker $USER

# 注销并重新登录，或运行以下命令使更改生效
newgrp docker

# 验证安装
docker --version
docker run hello-world
```

### 方法 2：手动安装（推荐用于生产环境）

```bash
# 1. 更新包索引
sudo apt update

# 2. 安装必要的包
sudo apt install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# 3. 添加 Docker 的官方 GPG 密钥
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# 4. 设置 Docker 仓库
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 5. 更新包索引
sudo apt update

# 6. 安装 Docker Engine
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 7. 将用户添加到 docker 组
sudo usermod -aG docker $USER

# 8. 启动 Docker 服务
sudo systemctl start docker
sudo systemctl enable docker

# 9. 注销并重新登录，然后验证
docker --version
docker run hello-world
```

### 方法 3：使用 apt（最快）

```bash
# 安装 Docker
sudo apt update
sudo apt install -y docker.io docker-compose

# 启动 Docker
sudo systemctl start docker
sudo systemctl enable docker

# 添加用户到 docker 组
sudo usermod -aG docker $USER

# 注销并重新登录，或运行
newgrp docker

# 验证
docker --version
```

---

## ✅ 验证安装

安装完成后，运行以下命令验证：

```bash
# 检查 Docker 版本
docker --version

# 检查 Docker Compose 版本
docker-compose --version

# 运行测试容器
docker run hello-world

# 检查 Docker 服务状态
sudo systemctl status docker
```

如果看到 "Hello from Docker!"，说明安装成功！

---

## 🔧 安装后配置

### 允许非 root 用户运行 Docker

```bash
# 添加当前用户到 docker 组
sudo usermod -aG docker $USER

# 应用更改（选择其一）
# 方法 1: 注销并重新登录
# 方法 2: 运行以下命令
newgrp docker

# 验证（不需要 sudo）
docker run hello-world
```

### 配置 Docker 开机自启

```bash
sudo systemctl enable docker
```

---

## 🚀 安装完成后

安装 Docker 后，返回项目目录并构建镜像：

```bash
cd /home/lu/streamlit/albums-main

# 构建 ALBuMS Docker 镜像
./build_complete_docker.sh
```

---

## 🆘 故障排除

### 问题 1：权限被拒绝

**错误**：`permission denied while trying to connect to the Docker daemon socket`

**解决方案**：
```bash
# 添加用户到 docker 组
sudo usermod -aG docker $USER

# 注销并重新登录，或运行
newgrp docker
```

### 问题 2：Docker 服务未运行

**错误**：`Cannot connect to the Docker daemon`

**解决方案**：
```bash
# 启动 Docker 服务
sudo systemctl start docker

# 设置开机自启
sudo systemctl enable docker
```

### 问题 3：端口已被占用

**错误**：`port is already allocated`

**解决方案**：
```bash
# 查看占用端口的进程
sudo lsof -i :8501

# 或使用不同的端口
docker run -p 8502:8501 albums-streamlit:latest
```

---

## 📚 更多资源

- [Docker 官方文档](https://docs.docker.com/engine/install/ubuntu/)
- [Docker Compose 文档](https://docs.docker.com/compose/)
- [Docker Hub](https://hub.docker.com/)

---

## 💡 快速命令参考

```bash
# 查看运行的容器
docker ps

# 查看所有容器
docker ps -a

# 停止容器
docker stop <container_id>

# 删除容器
docker rm <container_id>

# 查看镜像
docker images

# 删除镜像
docker rmi <image_id>

# 查看日志
docker logs <container_id>

# 进入容器
docker exec -it <container_id> /bin/bash
```

---

**安装完成后，运行 `./build_complete_docker.sh` 开始构建 ALBuMS 镜像！**
