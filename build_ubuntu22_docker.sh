#!/bin/bash
# DRFB Ubuntu 22 Docker 镜像构建和导出脚本

set -e

echo "========================================"
echo "DRFB - Ubuntu 22 Docker 镜像构建"
echo "========================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

# 检查必要的目录
echo -e "${BLUE}Step 1: 检查依赖目录${NC}"

if [ ! -d "mbtrack2-stable" ]; then
    echo -e "${RED}错误: mbtrack2-stable 目录不存在${NC}"
    echo "请确保 mbtrack2-stable 目录存在于当前目录中。"
    exit 1
fi

if [ ! -d "collective_effects" ]; then
    echo -e "${RED}错误: collective_effects 目录不存在${NC}"
    echo "请确保 collective_effects 目录存在于当前目录中。"
    exit 1
fi

echo -e "${GREEN}✓ 所有依赖目录已找到${NC}"
echo ""

# 选择 Dockerfile
echo -e "${BLUE}Step 2: 选择构建方法${NC}"
echo "1) Ubuntu 22 优化版本 (Dockerfile.ubuntu22) - 推荐"
echo "2) Python 3.10 基础版本 (Dockerfile.local)"
read -p "选择 (1 或 2, 默认 1): " choice
choice=${choice:-1}

if [ "$choice" = "1" ]; then
    DOCKERFILE="Dockerfile.ubuntu22"
    echo -e "${GREEN}使用 Ubuntu 22 优化版本构建${NC}"
else
    DOCKERFILE="Dockerfile.local"
    echo -e "${YELLOW}使用 Python 3.10 基础版本构建${NC}"
fi
echo ""

# 构建镜像
echo -e "${BLUE}Step 3: 构建 Docker 镜像${NC}"
IMAGE_NAME="albums-streamlit-ubuntu22"
IMAGE_TAG="latest"

echo "构建镜像: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "这可能需要几分钟..."
echo ""

if docker build -f "$DOCKERFILE" -t "${IMAGE_NAME}:${IMAGE_TAG}" .; then
    echo ""
    echo -e "${GREEN}✓ Docker 镜像构建成功!${NC}"
else
    echo ""
    echo -e "${RED}✗ Docker 镜像构建失败${NC}"
    exit 1
fi
echo ""

# 显示镜像信息
echo -e "${BLUE}Step 4: 镜像信息${NC}"
docker images "${IMAGE_NAME}:${IMAGE_TAG}"
echo ""

# 询问是否导出
read -p "导出镜像为 tar 文件以便分享? (y/n): " export_choice
if [ "$export_choice" = "y" ]; then
    echo ""
    echo -e "${BLUE}Step 5: 导出镜像${NC}"
    
    # 生成文件名（带日期）
    DATE=$(date +%Y%m%d)
    OUTPUT_FILE="albums-streamlit-ubuntu22-${DATE}.tar"
    
    echo "导出镜像到: ${OUTPUT_FILE}"
    echo "这可能需要几分钟..."
    
    if docker save -o "${OUTPUT_FILE}" "${IMAGE_NAME}:${IMAGE_TAG}"; then
        echo ""
        echo -e "${GREEN}✓ 镜像导出成功${NC}"
        
        # 显示文件大小
        FILE_SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
        echo "文件: ${OUTPUT_FILE}"
        echo "大小: ${FILE_SIZE}"
        echo ""
        
        # 压缩镜像
        read -p "压缩镜像文件? (y/n): " compress_choice
        if [ "$compress_choice" = "y" ]; then
            echo "压缩中..."
            if gzip "${OUTPUT_FILE}"; then
                COMPRESSED_FILE="${OUTPUT_FILE}.gz"
                COMPRESSED_SIZE=$(du -h "${COMPRESSED_FILE}" | cut -f1)
                echo -e "${GREEN}✓ 压缩完成${NC}"
                echo "压缩文件: ${COMPRESSED_FILE}"
                echo "压缩大小: ${COMPRESSED_SIZE}"
                echo ""
                echo -e "${YELLOW}分享方式:${NC}"
                echo "1. 发送 ${COMPRESSED_FILE} 给其他用户"
                echo "2. 其他用户运行:"
                echo "   gunzip ${COMPRESSED_FILE}"
                echo "   docker load -i ${OUTPUT_FILE}"
                echo "   docker run -p 8501:8501 ${IMAGE_NAME}:${IMAGE_TAG}"
            fi
        fi
    else
        echo ""
        echo -e "${RED}✗ 镜像导出失败${NC}"
        exit 1
    fi
fi
echo ""

# 询问是否测试镜像
read -p "现在测试镜像? (y/n): " test_choice
if [ "$test_choice" = "y" ]; then
    echo ""
    echo -e "${BLUE}Step 6: 测试镜像${NC}"
    echo "启动容器... (按 Ctrl+C 停止)"
    echo ""
    docker run -p 8501:8501 "${IMAGE_NAME}:${IMAGE_TAG}"
fi

echo -e "${GREEN}✓ 完成!${NC}"
