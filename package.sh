#!/bin/bash
# 简单的项目打包脚本

echo "正在打包 DRFB 项目..."

cd /home/lu/streamlit

# 打包项目，排除不需要的文件
tar -czf DRFB.tar.gz DRFB/ \
    --exclude="DRFB/.venv" \
    --exclude="DRFB/.gemini" \
    --exclude="DRFB/__pycache__" \
    --exclude="DRFB/**/__pycache__" \
    --exclude="DRFB/.git" \
    --exclude="DRFB/.vscode" \
    --exclude="DRFB/data" \
    --exclude="DRFB/*.pyc" \
    --exclude="DRFB/**/*.pyc" \
    --exclude="DRFB/*.tar.gz"

echo ""
echo "打包完成！"
echo "文件: /home/lu/streamlit/DRFB.tar.gz"
ls -lh /home/lu/streamlit/DRFB.tar.gz
echo ""
echo "使用方法："
echo "1. 解压: tar -xzf DRFB.tar.gz"
echo "2. 进入目录: cd DRFB"
echo "3. 安装依赖: pip install -r requirements.txt && pip install -r requirements_streamlit.txt"
echo "4. 运行: streamlit run streamlit_app.py"
echo ""
