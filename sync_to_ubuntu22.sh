#!/bin/bash
# WSL to Ubuntu 22 文件传输脚本

set -e

echo "========================================="
echo "DRFB WSL → Ubuntu 22 文件传输"
echo "========================================="
echo ""

# 配置
SSH_SERVER="172.28.3.95"
SSH_USER="rf"
TARGET_DIR="/home/rf/streamlit/DRFB"
SOURCE_DIR="/home/lu/streamlit/DRFB"

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}配置信息:${NC}"
echo "  源目录: $SOURCE_DIR"
echo "  目标服务器: $SSH_SERVER"
echo "  目标用户: $SSH_USER"
echo "  目标目录: $TARGET_DIR"
echo ""

# Step 1: 检查源目录
echo -e "${BLUE}Step 1: 检查源目录${NC}"
if [ ! -d "$SOURCE_DIR" ]; then
    echo -e "${RED}错误: 源目录不存在${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 源目录已找到${NC}"
echo ""

# Step 2: 检查 SSH 连接
echo -e "${BLUE}Step 2: 检查 SSH 连接${NC}"
if ssh -o ConnectTimeout=5 "$SSH_USER@$SSH_SERVER" "echo '✓ SSH 连接成功'" 2>/dev/null; then
    echo -e "${GREEN}✓ SSH 连接正常${NC}"
else
    echo -e "${RED}✗ SSH 连接失败${NC}"
    echo "请检查:"
    echo "  1. 服务器地址是否正确: $SSH_SERVER"
    echo "  2. 用户名是否正确: $SSH_USER"
    echo "  3. 网络连接是否正常"
    exit 1
fi
echo ""

# Step 3: 压缩文件
echo -e "${BLUE}Step 3: 压缩项目文件${NC}"
BACKUP_FILE="drfb-wsl-backup-$(date +%Y%m%d_%H%M%S).tar.gz"
echo "压缩文件: $BACKUP_FILE"
echo "这可能需要几分钟..."

cd /home/lu/streamlit
tar --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='*.egg-info' \
    --exclude='.streamlit/cache' \
    -czf "$BACKUP_FILE" DRFB/

if [ ! -f "$BACKUP_FILE" ]; then
    echo -e "${RED}✗ 文件压缩失败${NC}"
    exit 1
fi

FILE_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo -e "${GREEN}✓ 文件压缩成功${NC}"
echo "  文件大小: $FILE_SIZE"
echo ""

# Step 4: 创建目标目录
echo -e "${BLUE}Step 4: 创建远程目录${NC}"
ssh "$SSH_USER@$SSH_SERVER" "mkdir -p '$TARGET_DIR'" 2>/dev/null
echo -e "${GREEN}✓ 远程目录已创建${NC}"
echo ""

# Step 5: 传输文件
echo -e "${BLUE}Step 5: 传输文件到 Ubuntu 22${NC}"
echo "这可能需要几分钟（取决于网络速度）..."

if scp "$BACKUP_FILE" "$SSH_USER@$SSH_SERVER:$TARGET_DIR/"; then
    echo -e "${GREEN}✓ 文件传输成功${NC}"
else
    echo -e "${RED}✗ 文件传输失败${NC}"
    exit 1
fi
echo ""

# Step 6: 远程解压
echo -e "${BLUE}Step 6: 在远程服务器上解压${NC}"

ssh "$SSH_USER@$SSH_SERVER" "cd '$TARGET_DIR' && \
    tar -xzf '$(basename $BACKUP_FILE)' && \
    rm '$(basename $BACKUP_FILE)' && \
    echo '✓ 解压完成' && \
    ls -la"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 远程解压成功${NC}"
else
    echo -e "${RED}✗ 远程解压失败${NC}"
    exit 1
fi
echo ""

# Step 7: 验证
echo -e "${BLUE}Step 7: 验证文件${NC}"
echo "检查关键文件是否存在..."

FILES_TO_CHECK=(
    "streamlit_app.py"
    "requirements.txt"
    "Dockerfile.ubuntu22"
    "Makefile"
    "pages/"
    "utils/"
    "albums/"
)

MISSING=0
for file in "${FILES_TO_CHECK[@]}"; do
    if ssh "$SSH_USER@$SSH_SERVER" "test -e '$TARGET_DIR/DRFB/$file'" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${RED}✗${NC} $file"
        MISSING=$((MISSING+1))
    fi
done

if [ $MISSING -eq 0 ]; then
    echo -e "${GREEN}✓ 所有关键文件已同步${NC}"
else
    echo -e "${YELLOW}⚠ 有 $MISSING 个文件缺失${NC}"
fi
echo ""

# Step 8: 清理本地备份
read -p "删除本地备份文件? (y/n): " cleanup_choice
if [ "$cleanup_choice" = "y" ]; then
    rm "$BACKUP_FILE"
    echo -e "${GREEN}✓ 本地备份已删除${NC}"
else
    echo "本地备份保留: $BACKUP_FILE"
fi
echo ""

# 完成
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}✓ 文件传输完成!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "后续步骤:"
echo "  1. SSH 连接到 Ubuntu 22:"
echo "     ssh $SSH_USER@$SSH_SERVER"
echo ""
echo "  2. 进入项目目录:"
echo "     cd $TARGET_DIR/DRFB"
echo ""
echo "  3. 启动应用:"
echo "     ./start_ubuntu22.sh"
echo ""
echo "  或使用快速命令:"
echo "     make docker-dev"
echo ""
echo "  或查看所有命令:"
echo "     make help"
echo ""
