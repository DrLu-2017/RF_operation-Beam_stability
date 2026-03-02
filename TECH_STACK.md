# 🏗️ 技术栈和架构

## 应用架构

```
┌─────────────────────────────────────┐
│   Streamlit Web UI                  │
│   (streamlit_app.py + pages/)       │
└──────────────┬──────────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼────────┐    ┌──────▼────────┐
│   Utils    │    │    Albums     │
│ (RF calc)  │    │ (Core Logic)  │
└────┬───────┘    └──────┬────────┘
     │                   │
     └───────────┬───────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼──────────┐    ┌────────▼─────┐
│  mbtrack2    │    │  pycolleff    │
│ (Particle)   │    │(Collective Fx)│
└──────────────┘    └───────────────┘
```

---

## 核心依赖

| 库 | 版本 | 用途 |
|----|------|------|
| **Streamlit** | >=1.30.0 | Web UI 框架 |
| **Plotly** | >=5.18.0 | 交互式图表 |
| **Pandas** | >=2.0.0 | 数据处理 |
| **NumPy** | >=1.24.0 | 数值计算 |
| **SciPy** | >=1.11.0 | 科学算法 |
| **Matplotlib** | >=3.8.0 | 静态图表 |
| **Seaborn** | >=0.13.0 | 统计图表 |
| **mbtrack2** | v0.9.1 | 粒子追踪库 |
| **pycolleff** | v0.3.0 | 集合效应库 |

---

## 文件结构详解

```
/home/lu/streamlit/DRFB/
│
├── 🎯 主应用
│   ├── streamlit_app.py         # 应用入口、主菜单
│   └── .streamlit/              # Streamlit 配置
│
├── 📄 多页面应用 (pages/)
│   ├── 0_🔧_Double_RF_System.py    # RF 系统配置
│   ├── 1_📈_Semi_Analytic.py        # 半解析模型
│   ├── 2_🚀_MBTrack2_Remote.py       # 粒子追踪
│   └── 3_👥_User_Management.py      # 用户管理
│
├── 🛠️ 工具库 (utils/)
│   ├── rf_calculations.py       # RF 计算
│   ├── rf_calibration.py        # RF 标定
│   ├── config_manager.py        # 配置管理
│   ├── visualization.py         # 可视化工具
│   ├── [models].py              # 物理模型
│   │   ├── bosch_model.py
│   │   ├── alves_model.py
│   │   └── ...
│   └── ui_utils.py              # UI 工具
│
├── 💾 核心库 (albums/)
│   ├── scan.py                  # 扫描功能
│   ├── saveload.py              # 保存/加载
│   ├── optimiser.py             # 优化器
│   ├── robinson.py              # Robinson 模型
│   ├── mbtrack2_to_pycolleff.py # 库转换
│   └── plot_func.py             # 绘图函数
│
├── 📚 RF 系统 (rf_system/)
│   ├── rf_system_pro.py         # 专业 RF 系统
│   ├── rf_calc_app.py           # RF 计算应用
│   ├── rf_calc_base.py          # RF 基础计算
│   └── clbi_calculator.py       # CLBI 计算器
│
├── 📝 文档
│   ├── README.md                # 项目说明
│   ├── INSTALLATION_GUIDE.md    # 安装指南
│   ├── UBUNTU_22_SETUP.md       # Ubuntu 22 设置
│   ├── DOCKER_QUICKSTART.md     # Docker 快速开始
│   ├── DOCKER_GUIDE.md          # Docker 详细指南
│   ├── QUICK_REFERENCE.md       # 快速参考
│   └── docs/                    # Sphinx 文档
│
├── 🐳 Docker 配置
│   ├── Dockerfile              # 网络下载版
│   ├── Dockerfile.local        # 本地依赖版
│   ├── Dockerfile.ubuntu22     # Ubuntu 22 优化版
│   ├── docker-compose.yml      # 生产配置
│   ├── docker-compose.dev.yml  # 开发配置
│   ├── build_complete_docker.sh   # 完整构建脚本
│   └── build_ubuntu22_docker.sh   # Ubuntu 22 脚本
│
├── 🚀 启动脚本
│   ├── start_app.sh            # Linux/Mac 启动
│   ├── start_ubuntu22.sh       # Ubuntu 22 专用
│   ├── run.bat                 # Windows 启动
│   └── install.ps1             # Windows 安装
│
├── ⚙️ 配置文件
│   ├── Makefile                # 快速命令
│   ├── .env.example            # 环境变量模板
│   ├── .vscode/                # VS Code 配置
│   ├── .devcontainer/          # 开发容器配置
│   └── requirements.txt        # 依赖列表
│
├── 🧪 测试
│   ├── tests/                  # 测试代码
│   │   ├── test_unified_config.py
│   │   ├── test_preset_sync.py
│   │   └── ...
│   └── test_presets.py         # 预设测试
│
├── 📊 示例
│   ├── examples/
│   │   ├── Single_RF_instabilities.ipynb
│   │   ├── SOLEIL_II_instability_map.ipynb
│   │   └── SOLEIL_II_optimize_RoQ_Q0.ipynb
│   └── static/                 # 静态资源
│
├── 📦 依赖项
│   ├── mbtrack2-stable/        # 粒子追踪库
│   └── collective_effects/     # 集合效应库
│       └── pycolleff/
│
└── 其他
    ├── auth_config.yaml        # 认证配置
    ├── ssh_profiles.json       # SSH 配置
    └── pyproject.toml          # 项目元数据
```

---

## 开发环境拓扑

### 本地开发
```
开发者机器 (Ubuntu 22)
└── Python 3.10 虚拟环境
    ├── Streamlit (Web UI)
    ├── pandas/scipy/numpy (数据处理)
    ├── mbtrack2 (粒子追踪)
    └── pycolleff (集合效应)
    │
    └── 访问: http://localhost:8501
```

### Docker 开发
```
开发者机器 (任意操作系统)
└── Docker 容器
    ├── Ubuntu 22.04 基础镜像
    ├── Python 3.10
    └── 所有依赖库
    │
    └── 访问: http://localhost:8502 (开发) 或 8501 (生产)
```

---

## Python 版本要求

| 版本 | 状态 | 注意 |
|------|------|------|
| 3.10 | ✅ 推荐 | 最佳兼容性 |
| 3.11 | ✅ 支持 | 可能需要调整 |
| 3.12+ | ⚠️ 测试 | 部分库可能不兼容 |
| 3.9- | ❌ 不支持 | 依赖需要更高版本 |

---

## Ubuntu 22 特定配置

### 系统包
```bash
python3.10 python3-pip python3-venv
build-essential libopenblas-dev liblapack-dev gfortran
git curl docker.io docker-compose
```

### Python Wheels
预编译的 numpy/scipy 轮可通过 apt 在 Ubuntu 22 上获得，加快安装速度。

---

## Docker 镜像大小参考

| 镜像类型 | 未压缩 | 压缩后 |
|---------|--------|--------|
| Dockerfile (网络) | ~3-4 GB | ~1.5 GB |
| Dockerfile.local | ~2-3 GB | ~800 MB |
| Dockerfile.ubuntu22 | ~2 GB | ~700 MB |

---

## 性能优化建议

### 开发环境
- 使用 Docker 开发模式进行实时编辑
- 虚拟环境本地运行最快迭代

### 生产环境
- 使用多阶段 Dockerfile 减小镜像大小
- 启用 Streamlit 缓存机制
- 配置适当的 memory limits

---

## 集成工具

| 工具 | 用途 |
|------|------|
| **pytest** | 单元测试 |
| **flake8** | 代码风格检查 |
| **black** | 代码自动格式化 |
| **pylint** | 代码质量分析 |
| **VS Code** | 代码编辑 |
| **Docker** | 容器化 |
| **Docker Compose** | 多容器编排 |

---

## 相关文档链接

-👉 [快速参考](QUICK_REFERENCE.md)
- 👉 [Ubuntu 22 设置](UBUNTU_22_SETUP.md)
- 👉 [Docker 快速开始](DOCKER_QUICKSTART.md)
- 👉 [完整安装指南](INSTALLATION_GUIDE.md)
