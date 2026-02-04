# 🐳 Docker 完整镜像快速开始

## 🎯 目标

创建一个包含所有依赖的 Docker 镜像，其他人只需运行 Docker 就可以使用 ALBuMS，无需安装 mbtrack2 和 pycolleff。

---

## ⚡ 快速开始（3 步）

### 步骤 1：构建镜像

```bash
cd /home/lu/streamlit/albums-main
./build_complete_docker.sh
```

按提示操作：
- 选择 "1" (使用本地依赖)
- 选择 "y" (导出镜像)
- 选择 "y" (压缩镜像)

### 步骤 2：测试镜像

```bash
docker run -p 8501:8501 albums-streamlit:latest
```

打开浏览器访问 http://localhost:8501

### 步骤 3：分享

将生成的 `albums-streamlit-complete-YYYYMMDD.tar.gz` 文件分享给其他人。

---

## 📦 镜像包含的内容

✅ **完整的计算环境**：
- Python 3.10
- mbtrack2 (v0.9.1) - 粒子追踪库
- pycolleff (v0.3.0) - 集体效应库
- 所有 Python 依赖

✅ **ALBuMS 应用**：
- Streamlit UI
- 所有页面和功能
- 预设配置
- 示例文件

✅ **即开即用**：
- 无需安装任何依赖
- 无需配置环境
- 一条命令启动

---

## 📊 预期大小

- **Docker 镜像**: ~2-3 GB
- **压缩后 tar.gz**: ~800 MB - 1.2 GB

---

## 🚀 接收者如何使用

### 方法 1：使用 tar.gz 文件

```bash
# 1. 解压
gunzip albums-streamlit-complete-20260201.tar.gz

# 2. 加载镜像
docker load -i albums-streamlit-complete-20260201.tar

# 3. 运行
docker run -p 8501:8501 albums-streamlit:latest

# 4. 访问 http://localhost:8501
```

### 方法 2：使用 Docker Hub（如果你上传了）

```bash
# 直接运行
docker run -p 8501:8501 drlu2017/albums-streamlit:latest
```

---

## 📋 完整文档

- **[DOCKER_GUIDE.md](DOCKER_GUIDE.md)** - 详细的 Docker 指南
- **[README.md](README.md)** - 项目说明
- **[INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)** - 本地安装指南

---

## 🔍 验证镜像

构建后，验证镜像包含所有依赖：

```bash
docker run -it albums-streamlit:latest /bin/bash

# 在容器中测试
python -c "import mbtrack2; print('✓ mbtrack2')"
python -c "from pycolleff.longitudinal_equilibrium import LongitudinalEquilibrium; print('✓ pycolleff')"
python -c "from albums.robinson import RobinsonModes; print('✓ ALBuMS')"

exit
```

---

## 💡 提示

1. **首次构建**需要 10-20 分钟（取决于网络速度）
2. **后续构建**会快很多（Docker 缓存）
3. **压缩镜像**可以减少 50-70% 的文件大小
4. **使用 Docker Hub** 分享最方便（无需传输大文件）

---

## 🆘 需要帮助？

查看 [DOCKER_GUIDE.md](DOCKER_GUIDE.md) 获取详细说明和故障排除。
