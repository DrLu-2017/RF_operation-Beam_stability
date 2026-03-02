.PHONY: help install setup dev prod build test clean docker-up docker-down docker-build

help:
	@echo "DRFB - Ubuntu 22 开发环境"
	@echo ""
	@echo "可用命令:"
	@echo "  make install            - 安装 Python 依赖"
	@echo "  make setup              - 完整设置（虚拟环境 + 依赖）"
	@echo "  make dev                - 启动本地开发服务器"
	@echo "  make docker-dev         - 使用 Docker 开发模式运行"
	@echo "  make docker-prod        - 使用 Docker 生产模式运行"
	@echo "  make docker-build       - 构建 Ubuntu 22 Docker 镜像"
	@echo "  make test               - 运行测试"
	@echo "  make lint               - 代码风格检查"
	@echo "  make clean              - 清理临时文件"

install:
	@echo "安装 Python 依赖..."
	pip install -r requirements.txt
	pip install -r requirements_streamlit.txt

setup:
	@echo "设置开发环境..."
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip setuptools wheel
	$(MAKE) install
	@echo "✓ 环境设置完成"

dev:
	@echo "启动 Streamlit 开发服务器..."
	. .venv/bin/activate && streamlit run streamlit_app.py --server.port=8501

docker-dev:
	@echo "启动 Docker 开发模式..."
	docker-compose -f docker-compose.dev.yml up albums-dev

docker-prod:
	@echo "启动 Docker 生产模式..."
	docker-compose up albums-prod

docker-build:
	@echo "构建 Ubuntu 22 Docker 镜像..."
	./build_ubuntu22_docker.sh

test:
	@echo "运行测试..."
	. .venv/bin/activate && pytest tests/ -v

lint:
	@echo "检查代码风格..."
	. .venv/bin/activate && flake8 . --exclude=.venv,mbtrack2-stable,collective_effects,venv

format:
	@echo "格式化代码..."
	. .venv/bin/activate && black . --exclude=.venv,mbtrack2-stable,collective_effects,venv

clean:
	@echo "清理临时文件..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
	@echo "✓ 清理完成"

docker-clean:
	@echo "清理 Docker..."
	docker system prune -f
	@echo "✓ Docker 清理完成"

all: clean setup test
	@echo "✓ 完整设置和测试完成"
