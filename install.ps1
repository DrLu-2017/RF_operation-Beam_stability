#!/usr/bin/env pwsh
# ALBuMS Streamlit Installation Script for PowerShell

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  ALBuMS Streamlit Application - Setup" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Get the script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Check if .venv exists
$venvPath = Join-Path $ScriptDir ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "错误: 虚拟环境不存在: $venvPath" -ForegroundColor Red
    Write-Host ""
    Write-Host "请先创建虚拟环境:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv"
    Write-Host ""
    Read-Host "按 Enter 键退出"
    exit 1
}

# Activate virtual environment
Write-Host "正在激活虚拟环境..." -ForegroundColor Yellow
$activateScript = Join-Path $venvPath "Scripts" "Activate.ps1"

if (-not (Test-Path $activateScript)) {
    Write-Host "错误: 无法找到激活脚本: $activateScript" -ForegroundColor Red
    Read-Host "按 Enter 键退出"
    exit 1
}

& $activateScript

Write-Host "✓ 虚拟环境已激活" -ForegroundColor Green
Write-Host ""

# Upgrade pip
Write-Host "正在更新 pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip | Out-Null

Write-Host ""

# Install requirements
Write-Host "正在安装项目依赖..." -ForegroundColor Yellow
Write-Host "这可能需要几分钟..." -ForegroundColor Gray
Write-Host ""

pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "错误: 安装依赖失败" -ForegroundColor Red
    Write-Host ""
    Write-Host "如果网络连接超时，尝试使用国内镜像:" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/" -ForegroundColor Cyan
    Write-Host "或" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple" -ForegroundColor Cyan
    Write-Host ""
    Read-Host "按 Enter 键退出"
    exit 1
}

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "✓ 安装完成！" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "现在可以运行 Streamlit 应用:" -ForegroundColor Yellow
Write-Host "  streamlit run streamlit_app.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "应用将在浏览器中打开: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
