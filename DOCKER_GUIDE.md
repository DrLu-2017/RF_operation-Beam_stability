# Docker 完整镜像构建和分享指南

本指南说明如何构建包含所有依赖（mbtrack2、pycolleff）的完整 Docker 镜像，以便分享给其他人使用。

## 📋 准备工作

确保以下目录存在于项目根目录：
- ✅ `mbtrack2-stable/` - mbtrack2 库
- ✅ `collective_effects/` - pycolleff 库

这些目录已经在你的本地环境中，会被包含在 Docker 镜像中。

---

## 🚀 方法 1：使用自动化脚本（推荐）

### 步骤 1：运行构建脚本

```bash
cd /home/lu/streamlit/albums-main
./build_complete_docker.sh
```

脚本会：
1. 检查依赖目录
2. 构建 Docker 镜像
3. 可选：导出镜像为 tar 文件
4. 可选：压缩镜像文件

### 步骤 2：测试镜像

```bash
# 运行容器
docker run -p 8501:8501 albums-streamlit:latest

# 访问应用
# 打开浏览器访问 http://localhost:8501
```

### 步骤 3：分享镜像

如果你选择了导出镜像，会生成类似这样的文件：
- `albums-streamlit-complete-20260201.tar.gz` (压缩版)
- 或 `albums-streamlit-complete-20260201.tar` (未压缩)

**分享给其他人**：
1. 将 tar.gz 文件发送给接收者
2. 接收者运行：
   ```bash
   # 解压（如果是 .gz 文件）
   gunzip albums-streamlit-complete-20260201.tar.gz
   
   # 加载镜像
   docker load -i albums-streamlit-complete-20260201.tar
   
   # 运行应用
   docker run -p 8501:8501 albums-streamlit:latest
   
   # 访问 http://localhost:8501
   ```

---

## 🔧 方法 2：手动构建

### 使用本地依赖（推荐）

```bash
# 构建镜像
docker build -f Dockerfile.local -t albums-streamlit:latest .

# 运行
docker run -p 8501:8501 albums-streamlit:latest
```

### 从网络下载依赖

```bash
# 构建镜像（需要访问 GitLab 和 GitHub）
docker build -f Dockerfile -t albums-streamlit:latest .

# 运行
docker run -p 8501:8501 albums-streamlit:latest
```

---

## 📦 导出和压缩镜像

### 导出镜像

```bash
# 导出为 tar 文件
docker save -o albums-streamlit.tar albums-streamlit:latest

# 查看文件大小
du -h albums-streamlit.tar
```

### 压缩镜像（推荐用于分享）

```bash
# 压缩 tar 文件
gzip albums-streamlit.tar

# 这会创建 albums-streamlit.tar.gz
# 压缩后大小通常减少 50-70%
```

---

## 🌐 上传到 Docker Hub（可选）

如果你想通过 Docker Hub 分享：

### 步骤 1：登录 Docker Hub

```bash
docker login
# 输入你的 Docker Hub 用户名和密码
```

### 步骤 2：标记镜像

```bash
# 替换 yourusername 为你的 Docker Hub 用户名
docker tag albums-streamlit:latest yourusername/albums-streamlit:latest
```

### 步骤 3：推送到 Docker Hub

```bash
docker push yourusername/albums-streamlit:latest
```

### 步骤 4：其他人使用

其他人可以直接运行：
```bash
docker run -p 8501:8501 yourusername/albums-streamlit:latest
```

---

## 📊 镜像大小优化

### 当前镜像包含：
- ✅ Python 3.10
- ✅ Streamlit 和所有 UI 依赖
- ✅ mbtrack2 (粒子追踪库)
- ✅ pycolleff (集体效应库)
- ✅ 所有 Python 依赖
- ✅ ALBuMS 应用代码

### 预期大小：
- 未压缩镜像: ~2-3 GB
- 压缩后: ~800 MB - 1.2 GB

### 减小镜像大小的建议：
1. 使用 `.dockerignore` 排除不必要的文件
2. 使用多阶段构建（已在 Dockerfile 中实现）
3. 清理临时文件（已在 Dockerfile 中实现）

---

## 🔍 验证镜像

### 检查镜像是否包含所有依赖

```bash
# 运行容器并进入 shell
docker run -it albums-streamlit:latest /bin/bash

# 在容器中测试
python -c "import mbtrack2; print('mbtrack2:', mbtrack2.__version__)"
python -c "from pycolleff.longitudinal_equilibrium import LongitudinalEquilibrium; print('pycolleff: OK')"
python -c "from albums.robinson import RobinsonModes; print('ALBuMS: OK')"

# 退出
exit
```

---

## 📝 使用 docker-compose

创建 `docker-compose.yml`（已提供）：

```bash
# 启动
docker-compose up

# 后台运行
docker-compose up -d

# 停止
docker-compose down
```

---

## 🆘 故障排除

### 问题 1：构建失败 - 找不到 mbtrack2-stable

**解决方案**：
```bash
# 确保目录存在
ls -la mbtrack2-stable/
ls -la collective_effects/
```

### 问题 2：镜像太大

**解决方案**：
- 使用压缩：`gzip albums-streamlit.tar`
- 或使用 Docker Hub 分享（不需要传输文件）

### 问题 3：Docker 没有安装

**解决方案**：
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install docker.io docker-compose

# 添加用户到 docker 组
sudo usermod -aG docker $USER
# 注销并重新登录
```

---

## 📚 接收者使用指南

如果你要分享给其他人，给他们这个简单的指南：

### 使用 tar 文件

```bash
# 1. 解压（如果是 .gz 文件）
gunzip albums-streamlit-complete-YYYYMMDD.tar.gz

# 2. 加载镜像
docker load -i albums-streamlit-complete-YYYYMMDD.tar

# 3. 运行应用
docker run -p 8501:8501 albums-streamlit:latest

# 4. 打开浏览器访问
# http://localhost:8501
```

### 使用 Docker Hub

```bash
# 直接运行（会自动下载）
docker run -p 8501:8501 yourusername/albums-streamlit:latest

# 访问 http://localhost:8501
```

---

## ✅ 总结

**推荐的分享流程**：

1. **构建镜像**：
   ```bash
   ./build_complete_docker.sh
   ```

2. **选择分享方式**：
   - **文件分享**：导出并压缩 tar 文件
   - **Docker Hub**：推送到 Docker Hub

3. **提供给接收者**：
   - tar.gz 文件 + 使用说明
   - 或 Docker Hub 链接

4. **接收者使用**：
   - 加载镜像或从 Docker Hub 拉取
   - 运行容器
   - 访问应用

---

**需要帮助？** 查看项目的 GitHub Issues 或联系维护者。
