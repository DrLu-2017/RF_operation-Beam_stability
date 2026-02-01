#!/bin/bash
# ALBuMS GitHub 同步脚本

set -e

echo "================================"
echo "ALBuMS GitHub 同步脚本"
echo "================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查是否已经是 Git 仓库
if [ ! -d .git ]; then
    echo -e "${BLUE}步骤 1: 初始化 Git 仓库${NC}"
    git init
    echo -e "${GREEN}✓ Git 仓库已初始化${NC}"
    echo ""
    
    # 检查 Git 配置
    if ! git config user.name > /dev/null 2>&1; then
        echo -e "${YELLOW}请配置 Git 用户信息：${NC}"
        read -p "输入你的名字: " git_name
        read -p "输入你的邮箱: " git_email
        git config user.name "$git_name"
        git config user.email "$git_email"
        echo -e "${GREEN}✓ Git 用户信息已配置${NC}"
    fi
    echo ""
else
    echo -e "${GREEN}✓ Git 仓库已存在${NC}"
    echo ""
fi

# 显示状态
echo -e "${BLUE}步骤 2: 查看文件状态${NC}"
git status
echo ""

# 询问是否继续
read -p "是否添加所有文件到 Git? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "已取消"
    exit 0
fi

# 添加文件
echo ""
echo -e "${BLUE}步骤 3: 添加文件${NC}"
git add .
echo -e "${GREEN}✓ 文件已添加${NC}"
echo ""

# 提交
echo -e "${BLUE}步骤 4: 创建提交${NC}"
read -p "输入提交信息 (默认: Update ALBuMS application): " commit_msg
commit_msg=${commit_msg:-"Update ALBuMS application"}
git commit -m "$commit_msg"
echo -e "${GREEN}✓ 提交已创建${NC}"
echo ""

# 检查是否已配置远程仓库
if ! git remote | grep -q origin; then
    echo -e "${YELLOW}未检测到远程仓库${NC}"
    echo ""
    echo "请在 GitHub 上创建一个新仓库，然后输入仓库 URL："
    echo "格式: https://github.com/用户名/仓库名.git"
    read -p "GitHub 仓库 URL: " repo_url
    
    if [ -n "$repo_url" ]; then
        git remote add origin "$repo_url"
        echo -e "${GREEN}✓ 远程仓库已添加${NC}"
        echo ""
        
        # 推送到 GitHub
        echo -e "${BLUE}步骤 5: 推送到 GitHub${NC}"
        git branch -M main
        git push -u origin main
        echo -e "${GREEN}✓ 已推送到 GitHub${NC}"
    else
        echo -e "${RED}未提供仓库 URL，跳过推送${NC}"
        echo "稍后可以手动运行："
        echo "  git remote add origin https://github.com/用户名/仓库名.git"
        echo "  git branch -M main"
        echo "  git push -u origin main"
    fi
else
    # 推送到现有远程仓库
    echo -e "${BLUE}步骤 5: 推送到 GitHub${NC}"
    git push
    echo -e "${GREEN}✓ 已推送到 GitHub${NC}"
fi

echo ""
echo "================================"
echo -e "${GREEN}同步完成！${NC}"
echo "================================"
echo ""
echo "你的代码已同步到 GitHub！"
echo ""
