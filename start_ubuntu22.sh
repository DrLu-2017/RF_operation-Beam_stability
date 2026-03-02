#!/bin/bash
# DRFB Ubuntu 22 快速开始脚本
# 自动化 Docker 和虚拟环境设置

set -e

echo "========================================"
echo "DRFB - Ubuntu 22 快速开始"
echo "========================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

# 检查操作系统
echo -e "${BLUE}检查 Ubuntu 版本...${NC}"
if grep -q "22.04" /etc/os-release; then
    echo -e "${GREEN}✓ 检测到 Ubuntu 22.04${NC}"
else
    echo -e "${YELLOW}⚠ 检测到其他 Ubuntu 版本（推荐使用 Ubuntu 22.04）${NC}"
    read -p "继续? (y/n): " confirm
    if [ "$confirm" != "y" ]; then
        exit 0
    fi
fi
echo ""

# 步骤 1: 检查 Docker
echo -e "${BLUE}Step 1: 检查 Docker${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Docker 未安装。安装中...${NC}"
    sudo apt update
    sudo apt install -y docker.io docker-compose
    sudo usermod -aG docker $USER
    echo -e "${GREEN}✓ Docker 已安装${NC}"
    echo -e "${YELLOW}提示: 请重新登录以使用户组更改生效${NC}"
else
    echo -e "${GREEN}✓ Docker 已安装${NC}"
    docker --version
fi
echo ""

# 步骤 2: 检查 Python 虚拟环境
echo -e "${BLUE}Step 2: 设置 Python 虚拟环境${NC}"
if [ ! -d ".venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip setuptools wheel
    echo -e "${GREEN}✓ 虚拟环境已创建${NC}"
else
    echo -e "${GREEN}✓ 虚拟环境已存在${NC}"
    source .venv/bin/activate
fi
echo ""

# 步骤 3: 安装 Python 依赖
echo -e "${BLUE}Step 3: 安装 Python 依赖${NC}"
if pip install -q -r requirements.txt && pip install -q -r requirements_streamlit.txt; then
    echo -e "${GREEN}✓ Python 依赖已安装${NC}"
else
    echo -e "${RED}✗ 安装依赖失败${NC}"
    exit 1
fi
echo ""

# 步骤 4: 选择运行方式
echo -e "${BLUE}Step 4: 选择运行方式${NC}"
echo "1) Docker 开发模式 (推荐) - 支持实时代码更新"
echo "2) Docker 生产模式 - 完整隔离环境"
echo "3) 本地虚拟环境 - 直接运行"
read -p "选择 (1/2/3, 默认 1): " choice
choice=${choice:-1}
echo ""

case $choice in
    1)
        echo -e "${BLUE}启动 Docker 开发模式...${NC}"
        docker-compose -f docker-compose.dev.yml up albums-dev
        ;;
    2)
        echo -e "${BLUE}启动 Docker 生产模式...${NC}"
        docker-compose up albums-prod
        ;;
    3)
        echo -e "${BLUE}启动本地虚拟环境...${NC}"
        source .venv/bin/activate
        streamlit run streamlit_app.py --server.port=8501
        ;;
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac
