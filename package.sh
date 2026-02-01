#!/bin/bash
# 简单的项目打包脚本

echo "正在打包 ALBuMS 项目..."

cd /home/lu/streamlit

# 打包项目，排除不需要的文件
tar -czf albums-main.tar.gz albums-main/ \
    --exclude="albums-main/.venv" \
    --exclude="albums-main/.gemini" \
    --exclude="albums-main/__pycache__" \
    --exclude="albums-main/**/__pycache__" \
    --exclude="albums-main/.git" \
    --exclude="albums-main/.vscode" \
    --exclude="albums-main/data" \
    --exclude="albums-main/*.pyc" \
    --exclude="albums-main/**/*.pyc" \
    --exclude="albums-main/*.tar.gz"

echo ""
echo "打包完成！"
echo "文件: /home/lu/streamlit/albums-main.tar.gz"
ls -lh /home/lu/streamlit/albums-main.tar.gz
echo ""
echo "使用方法："
echo "1. 解压: tar -xzf albums-main.tar.gz"
echo "2. 进入目录: cd albums-main"
echo "3. 安装依赖: pip install -r requirements.txt && pip install -r requirements_streamlit.txt"
echo "4. 运行: streamlit run streamlit_app.py"
echo ""
