# ALBuMS - Advanced Longitudinal Beam Stability

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)

**ALBuMS** (Advanced Longitudinal Beam Stability) is a powerful Streamlit-based web application for analyzing and optimizing RF systems in particle accelerators, with a focus on double RF cavity configurations.

## ✨ Features

- 🔧 **Double RF System Analysis** - Interactive dashboard for configuring and analyzing main and harmonic cavity systems
- 📊 **Parameter Scans** - 2D stability maps across parameter spaces (ψ vs Current, ψ vs R/Q)
- 🎯 **R-Factor Optimization** - Maximize Touschek lifetime through cavity parameter optimization
- 🔬 **Mode Analysis** - Track Robinson modes and identify coupled-bunch instabilities
- 💾 **Configuration Management** - Save, load, and share accelerator configurations
- 🎨 **Interactive Visualization** - Dynamic plots with Plotly for exploring results
- 📦 **Preset Configurations** - Pre-configured settings for SOLEIL II, Aladdin, and more

## 🚀 Quick Start

### Option 1: Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/albums-streamlit.git
cd albums-streamlit

# Build and run with Docker
docker build -t albums .
docker run -p 8501:8501 albums

# Or use docker-compose
docker-compose up
```

Then open your browser to http://localhost:8501

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/albums-streamlit.git
cd albums-streamlit

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_streamlit.txt

# Install mbtrack2 (required for full functionality)
# See INSTALLATION_GUIDE.md for detailed instructions

# Run the application
streamlit run streamlit_app.py
```

## 📋 Requirements

### For UI Mode (Configuration Only)
- Python 3.10+
- Dependencies in `requirements.txt` and `requirements_streamlit.txt`

### For Full Mode (With Simulations)
- All UI mode requirements
- `mbtrack2` library (particle tracking)
- `pycolleff` library (collective effects)
- See `INSTALLATION_GUIDE.md` for installation instructions

## 📖 Documentation

- **[Installation Guide](INSTALLATION_GUIDE.md)** - Detailed installation instructions for full mode
- **[GitHub Sync Guide](GITHUB_SYNC_GUIDE.md)** - How to sync your changes to GitHub
- **[User Guide](docs/)** - Comprehensive user documentation

## 🎯 Usage

### 1. Double RF System Dashboard
Configure and analyze double RF cavity systems with:
- Main cavity parameters (voltage, frequency, R/Q, QL)
- Harmonic cavity parameters (voltage, harmonic multiplier, R/Q, QL)
- Machine parameters (energy, circumference, momentum compaction)

### 2. Parameter Scans
Perform 2D parameter scans to explore stability regions:
- **ψ vs Current**: Scan phase offset against beam current
- **ψ vs R/Q**: Scan phase offset against cavity R/Q

### 3. Optimization
Optimize cavity parameters to maximize:
- Touschek lifetime R-factor
- Bunch lengthening
- Stability margins

### 4. Mode Analysis
Analyze coupled-bunch modes:
- Robinson mode tracking
- Growth rate calculations
- Mode coupling identification

## 🏗️ Project Structure

```
albums-streamlit/
├── streamlit_app.py          # Main application entry point
├── pages/                    # Streamlit pages
│   ├── 0_🔧_Double_RF_System.py
│   ├── 1_📊_Parameter_Scans.py
│   ├── 2_🎯_Optimization.py
│   └── 3_🔬_Mode_Analysis.py
├── albums/                   # Core ALBuMS library
├── utils/                    # Utility functions
│   ├── presets.py           # Preset configurations
│   ├── config_manager.py    # Configuration management
│   └── visualization.py     # Plotting functions
├── examples/                 # Example configurations
├── tests/                    # Unit tests
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose setup
└── requirements.txt         # Python dependencies
```

## 🔬 Preset Configurations

The application includes pre-configured settings for:

- **SOLEIL II** - 4th generation synchrotron light source
  - Energy: 2.75 GeV
  - Main RF: 352.2 MHz
  - Harmonic RF: 1408.8 MHz (4th harmonic)

- **Aladdin** - Storage ring benchmark
  - Energy: 1.0 GeV
  - Main RF: 499.654 MHz
  - Harmonic RF: 1498.962 MHz (3rd harmonic)

- **Custom** - User-defined configurations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the BSD-3-Clause License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **mbtrack2** - Particle tracking library by SOLEIL Synchrotron
- **pycolleff** - Collective effects library by LNLS
- **Streamlit** - Web application framework

## 📧 Contact

For questions or support, please open an issue on GitHub.

## 📚 Citation

If you use ALBuMS in your research, please cite:

```bibtex
@software{albums_streamlit,
  title = {ALBuMS: Advanced Longitudinal Beam Stability Analysis},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/albums-streamlit}
}
```

---

**Made with ❤️ for accelerator physics**
