# 将 ALBuMS 项目同步到 GitHub 指南

## 步骤 1：初始化 Git 仓库

```bash
cd /home/lu/streamlit/albums-main

# 初始化 Git 仓库
git init

# 配置 Git 用户信息（如果还没配置）
git config user.name "你的名字"
git config user.email "你的邮箱@example.com"
```

## 步骤 2：添加文件到 Git

```bash
# 查看将要添加的文件
git status

# 添加所有文件（.gitignore 会自动排除不需要的文件）
git add .

# 查看暂存的文件
git status

# 创建第一次提交
git commit -m "Initial commit: ALBuMS Streamlit application"
```

## 步骤 3：在 GitHub 上创建仓库

1. **访问 GitHub**：https://github.com
2. **登录**你的 GitHub 账户
3. **点击右上角的 "+" 按钮** → 选择 "New repository"
4. **填写仓库信息**：
   - Repository name: `albums-streamlit` （或你喜欢的名字）
   - Description: `ALBuMS - Advanced Longitudinal Beam Stability Analysis`
   - 选择 **Public** 或 **Private**
   - **不要**勾选 "Initialize this repository with a README"
   - 点击 "Create repository"

## 步骤 4：连接本地仓库到 GitHub

GitHub 会显示一些命令，使用这些命令：

```bash
# 添加远程仓库（替换为你的 GitHub 用户名和仓库名）
git remote add origin https://github.com/你的用户名/albums-streamlit.git

# 验证远程仓库
git remote -v

# 推送到 GitHub（首次推送）
git branch -M main
git push -u origin main
```

## 步骤 5：后续更新

以后每次修改后，使用以下命令同步：

```bash
# 查看修改的文件
git status

# 添加修改的文件
git add .

# 提交修改
git commit -m "描述你的修改"

# 推送到 GitHub
git push
```

---

## 快速命令脚本

我已经为你准备了一个自动化脚本，运行以下命令即可：

```bash
cd /home/lu/streamlit/albums-main
./sync_to_github.sh
```

---

## 注意事项

### ✅ 会被提交的文件：
- 所有 Python 源代码（`.py` 文件）
- 配置文件（`requirements.txt`, `Dockerfile` 等）
- 文档（`README.md`, `*.md` 文件）
- 示例文件（`examples/` 目录）

### ❌ 不会被提交的文件（已在 .gitignore 中排除）：
- 虚拟环境（`.venv/`）
- Python 缓存（`__pycache__/`, `*.pyc`）
- 下载的依赖（`mbtrack2-stable/`, `collective_effects/`）
- 数据文件（`data/`, `*.h5`, `*.csv`）
- 临时文件（`*.tar.gz`, `.gemini/`）

---

## 常见问题

### Q: 如何查看将要提交的文件？
```bash
git status
```

### Q: 如何撤销某个文件的添加？
```bash
git reset HEAD 文件名
```

### Q: 如何查看提交历史？
```bash
git log --oneline
```

### Q: 如何克隆到另一台电脑？
```bash
git clone https://github.com/你的用户名/albums-streamlit.git
cd albums-streamlit
pip install -r requirements.txt
pip install -r requirements_streamlit.txt
```

---

## 推荐的 README.md 内容

你的项目已经有一个 README.md，但你可能想要添加：

- 项目截图
- 安装说明
- 使用示例
- 贡献指南
- 许可证信息

---

## 下一步

1. 运行 `./sync_to_github.sh` 或手动执行上述命令
2. 在 GitHub 上查看你的仓库
3. 可以添加 README badges、GitHub Actions 等
4. 分享给其他人！

---

## 需要帮助？

如果遇到问题，告诉我具体的错误信息，我可以帮你解决！
