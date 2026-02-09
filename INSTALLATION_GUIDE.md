# 安装 ALBuMS 完整模式指南

## 当前状态

你的 ALBuMS 应用目前运行在 **UI 模式**：
- ✅ 可以配置参数
- ✅ 可以保存/加载配置
- ✅ 可以使用预设
- ❌ 不能运行物理模拟

要启用**完整模式**（运行实际模拟），需要安装 `mbtrack2` 库。

---

## 安装选项

### 选项 1：从 GitLab 安装 mbtrack2（推荐）

`mbtrack2` 是 SOLEIL 同步辐射光源开发的粒子追踪库。

#### 步骤：

1. **检查是否有访问权限**
   ```bash
   # mbtrack2 通常托管在 SOLEIL 的 GitLab 上
   # 你可能需要访问权限
   ```

2. **克隆仓库**（如果有权限）
   ```bash
   cd /home/lu/streamlit/DRFB
   git clone https://gitlab.synchrotron-soleil.fr/pa/collective-effects/mbtrack2.git mbtrack2-stable
   ```

3. **安装依赖**
   ```bash
   source .venv/bin/activate
   cd mbtrack2-stable
   pip install -e .
   ```

---

### 选项 2：使用 Docker（最简单，推荐用于分享）

如果你主要是想分享给其他人使用，**强烈推荐使用 Docker**：

#### 优点：
- ✅ 不需要手动安装 mbtrack2
- ✅ 环境完全一致
- ✅ 接收者无需配置
- ✅ 已有 Dockerfile 配置

#### 步骤：

1. **安装 Docker**（如果还没安装）
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install docker.io docker-compose
   
   # 添加用户到 docker 组（避免每次都用 sudo）
   sudo usermod -aG docker $USER
   # 注销并重新登录使更改生效
   ```

2. **使用项目的 Dockerfile**
   
   你的项目已经有一个 Dockerfile，它基于 SOLEIL 的 mbtrack2 镜像：
   ```bash
   cd /home/lu/streamlit/DRFB
   docker build -t albums .
   docker run -p 8501:8501 albums
   ```

---

### 选项 3：仅用于演示/配置（当前模式）

如果你**不需要运行实际模拟**，只是想：
- 配置参数
- 保存配置
- 分享配置给其他人
- 教学演示

那么**当前的 UI 模式已经足够**，无需安装 mbtrack2。

---

## 推荐方案

### 如果你是为了自己使用：
**→ 选项 2（Docker）** - 最简单，最可靠

### 如果你是为了分享给其他人：
**→ 选项 2（Docker）** - 接收者只需运行 `docker run`

### 如果你只需要配置参数：
**→ 选项 3（当前模式）** - 已经可以使用，无需额外安装

---

## 快速决策

**问题 1：你需要运行实际的物理模拟吗？**
- 是 → 需要安装 mbtrack2（选项 1 或 2）
- 否 → 当前模式已足够（选项 3）

**问题 2：你有 SOLEIL GitLab 访问权限吗？**
- 是 → 可以使用选项 1
- 否 → 使用选项 2（Docker）

**问题 3：你主要是为了分享吗？**
- 是 → 强烈推荐选项 2（Docker）
- 否 → 根据需求选择

---

## Docker 快速开始（推荐）

```bash
# 1. 安装 Docker
sudo apt update
sudo apt install docker.io

# 2. 构建镜像
cd /home/lu/streamlit/DRFB
sudo docker build -t albums .

# 3. 运行应用
sudo docker run -p 8501:8501 albums

# 4. 访问应用
# 打开浏览器访问 http://localhost:8501
```

---

## 需要帮助？

如果你决定了使用哪个选项，告诉我，我可以帮你：
1. 安装 Docker
2. 构建 Docker 镜像
3. 或者配置其他安装方式
