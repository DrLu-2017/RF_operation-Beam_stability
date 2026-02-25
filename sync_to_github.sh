#!/bin/bash
# ALBuMS GitHub Sync Script

set -e

echo "================================"
echo "ALBuMS GitHub Sync Script"
echo "================================"
echo ""

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if it's already a Git repository
if [ ! -d .git ]; then
    echo -e "${BLUE}Step 1: Initializing Git Repository${NC}"
    git init
    echo -e "${GREEN}✓ Git repository initialized${NC}"
    echo ""
    
    # Check Git configuration
    if ! git config user.name > /dev/null 2>&1; then
        echo -e "${YELLOW}Please configure Git user information:${NC}"
        read -p "Enter your name: " git_name
        read -p "Enter your email: " git_email
        git config user.name "$git_name"
        git config user.email "$git_email"
        echo -e "${GREEN}✓ Git user info configured${NC}"
    fi
    echo ""
else
    echo -e "${GREEN}✓ Git repository already exists${NC}"
    echo ""
fi

# Show status
echo -e "${BLUE}Step 2: Checking File Status${NC}"
git status
echo ""

# Ask whether to continue
read -p "Add all files to Git? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "Cancelled"
    exit 0
fi

# Add files
echo ""
echo -e "${BLUE}Step 3: Staging Files${NC}"
git add .
echo -e "${GREEN}✓ Files staged${NC}"
echo ""

# Commit
echo -e "${BLUE}Step 4: Creating Commit${NC}"
read -p "Enter commit message (default: Update ALBuMS application): " commit_msg
commit_msg=${commit_msg:-"Update ALBuMS application"}
git commit -m "$commit_msg"
echo -e "${GREEN}✓ Commit created${NC}"
echo ""

# Check if a remote repository is configured
if ! git remote | grep -q origin; then
    echo -e "${YELLOW}No remote repository detected${NC}"
    echo ""
    echo "Please create a new repository on GitHub, then enter the repository URL:"
    echo "Format: https://github.com/Username/RepositoryName.git"
    read -p "GitHub Repository URL: " repo_url
    
    if [ -n "$repo_url" ]; then
        git remote add origin "$repo_url"
        echo -e "${GREEN}✓ Remote repository added${NC}"
        echo ""
        
        # Push to GitHub
        echo -e "${BLUE}Step 5: Pushing to GitHub${NC}"
        git branch -M main
        git push -u origin main
        echo -e "${GREEN}✓ Pushed to GitHub${NC}"
    else
        echo -e "${RED}No repository URL provided, skipping push${NC}"
        echo "You can run these manually later:"
        echo "  git remote add origin https://github.com/Username/RepositoryName.git"
        echo "  git branch -M main"
        echo "  git push -u origin main"
    fi
else
    # Push to existing remote repository
    echo -e "${BLUE}Step 5: Pushing to GitHub${NC}"
    git push
    echo -e "${GREEN}✓ Pushed to GitHub${NC}"
fi

echo ""
echo "================================"
echo -e "${GREEN}Sync Complete!${NC}"
echo "================================"
echo ""
echo "Your code has been synced to GitHub!"
echo ""
