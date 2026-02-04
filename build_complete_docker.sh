#!/bin/bash
# ALBuMS Docker 完整镜像构建脚本
# 包含 mbtrack2 和 pycolleff

set -e

echo "========================================"
echo "ALBuMS Docker 完整镜像构建脚本"
echo "========================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查必需的目录
echo -e "${BLUE}步骤 1: 检查依赖${NC}"

if [ ! -d "mbtrack2-stable" ]; then
    echo -e "${RED}错误: mbtrack2-stable 目录不存在${NC}"
    echo "请确保 mbtrack2-stable 目录存在于当前目录中"
    exit 1
fi

if [ ! -d "collective_effects" ]; then
    echo -e "${RED}错误: collective_effects 目录不存在${NC}"
    echo "请确保 collective_effects 目录存在于当前目录中"
    exit 1
fi

echo -e "${GREEN}✓ 所有依赖目录都存在${NC}"
echo ""

# 选择 Dockerfile
echo -e "${BLUE}步骤 2: 选择构建方式${NC}"
echo "1) 使用本地依赖 (Dockerfile.local) - 推荐"
echo "2) 从网络下载依赖 (Dockerfile)"
read -p "选择 (1 或 2, 默认 1): " choice
choice=${choice:-1}

if [ "$choice" = "1" ]; then
    DOCKERFILE="Dockerfile.local"
    echo -e "${GREEN}使用本地依赖构建${NC}"
else
    DOCKERFILE="Dockerfile"
    echo -e "${YELLOW}使用网络下载构建（可能需要访问权限）${NC}"
fi
echo ""

# 构建镜像
echo -e "${BLUE}步骤 3: 构建 Docker 镜像${NC}"
IMAGE_NAME="albums-streamlit"
IMAGE_TAG="latest"

echo "正在构建镜像: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "这可能需要几分钟..."
echo ""

if docker build -f "$DOCKERFILE" -t "${IMAGE_NAME}:${IMAGE_TAG}" .; then
    echo ""
    echo -e "${GREEN}✓ Docker 镜像构建成功！${NC}"
else
    echo ""
    echo -e "${RED}✗ Docker 镜像构建失败${NC}"
    exit 1
fi
echo ""

# 显示镜像信息
echo -e "${BLUE}步骤 4: 镜像信息${NC}"
docker images "${IMAGE_NAME}:${IMAGE_TAG}"
echo ""

# 询问是否导出
read -p "是否导出镜像为 tar 文件以便分享? (y/n): " export_choice
if [ "$export_choice" = "y" ]; then
    echo ""
    echo -e "${BLUE}步骤 5: 导出镜像${NC}"
    
    # 生成文件名（包含日期）
    DATE=$(date +%Y%m%d)
    OUTPUT_FILE="albums-streamlit-complete-${DATE}.tar"
    
    echo "正在导出镜像到: ${OUTPUT_FILE}"
    echo "这可能需要几分钟..."
    
    if docker save -o "${OUTPUT_FILE}" "${IMAGE_NAME}:${IMAGE_TAG}"; then
        echo ""
        echo -e "${GREEN}✓ 镜像已导出${NC}"
        
        # 显示文件大小
        FILE_SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
        echo "文件: ${OUTPUT_FILE}"
        echo "大小: ${FILE_SIZE}"
        echo ""
        
        # 压缩镜像
        read -p "是否压缩镜像文件? (y/n): " compress_choice
        if [ "$compress_choice" = "y" ]; then
            echo ""
            echo "正在压缩..."
            if gzip "${OUTPUT_FILE}"; then
                COMPRESSED_FILE="${OUTPUT_FILE}.gz"
                COMPRESSED_SIZE=$(du -h "${COMPRESSED_FILE}" | cut -f1)
                echo -e "${GREEN}✓ 镜像已压缩${NC}"
                echo "压缩文件: ${COMPRESSED_FILE}"
                echo "压缩后大小: ${COMPRESSED_SIZE}"
            else
                echo -e "${RED}压缩失败${NC}"
            fi
        fi
    else
        echo -e "${RED}✗ 镜像导出失败${NC}"
    fi
fi

echo ""
echo "========================================"
echo -e "${GREEN}构建完成！${NC}"
echo "========================================"
echo ""
echo "使用方法："
echo ""
echo "1. 本地运行："
echo "   docker run -p 8501:8501 ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "2. 使用 docker-compose："
echo "   docker-compose up"
echo ""
echo "3. 分享给其他人："
if [ -f "albums-streamlit-complete-${DATE}.tar.gz" ]; then
    echo "   分享文件: albums-streamlit-complete-${DATE}.tar.gz"
    echo "   接收者运行: gunzip albums-streamlit-complete-${DATE}.tar.gz"
    echo "              docker load -i albums-streamlit-complete-${DATE}.tar"
    echo "              docker run -p 8501:8501 ${IMAGE_NAME}:${IMAGE_TAG}"
elif [ -f "albums-streamlit-complete-${DATE}.tar" ]; then
    echo "   分享文件: albums-streamlit-complete-${DATE}.tar"
    echo "   接收者运行: docker load -i albums-streamlit-complete-${DATE}.tar"
    echo "              docker run -p 8501:8501 ${IMAGE_NAME}:${IMAGE_TAG}"
fi
echo ""
echo "访问应用: http://localhost:8501"
echo ""
