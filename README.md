# SOFI Dashboard - State of the Future Index

An interactive web dashboard for analyzing and simulating the State of the Future Index (SOFI) with multiple indicators across different dimensions of sustainable development.

![Dashboard Preview](https://img.shields.io/badge/Python-3.10-blue) ![Dash](https://img.shields.io/badge/Dash-3.2.0-green)

This dashboard can be accessed at https://vedanthirekarsofi.pythonanywhere.com/


## Features

### Historical Trends
- **SOFI Index Over Time**: Visualize the SOFI index progression from 1990 to 2035
- **Year-over-Year Changes**: Track percentage changes in SOFI across years
- **Individual Indicator Trends**: Compare multiple indicators with normalized and original value views

### What-If Simulator
- **Multi-Variable Simulation**: Adjust multiple indicators simultaneously
- **Three Growth Models**:
  - One-time change: Apply percentage once to all future years
  - Compound growth: Exponential year-over-year compounding
  - Annual rate: Linear growth/decline from baseline
- **Real-time SOFI Recalculation**: See immediate impact on SOFI index
- **Interactive Graphs**: Visual comparison of baseline vs adjusted scenarios

### Correlations
- **Scatter Plot Analysis**: Explore relationships between any two indicators
- **Correlation with SOFI**: Bar chart showing which indicators most strongly correlate with SOFI
- **Trend Lines**: Automatic regression lines for correlation visualization

### Indicator Analysis
- **Year Comparison**: Compare indicator values between any two years
- **Normalized vs Original Views**: Toggle between 0-1 normalized scale and original values
- **Grouped Bar Chart**: Side-by-side comparison of all 29 indicators


<!-- 
## 📁 Project Structure

```
SOFI-2035-Info-Viz-code/
│
├── viz.py                  # Main application file
├── data1.xlsx             # Data file with 3 sheets:
│                          #   - Sheet1: Normalized values (0-1)
│                          #   - Sheet2: Original values
│                          #   - Sheet3: Indicator weights
├── requirements.txt       # Python dependencies
├── wsgi.py               # WSGI configuration for deployment
└── README.md             # This file
``` -->
<!-- 
## 🌐 Deployment on PythonAnywhere

### Step-by-Step Guide

1. **Sign up** at [pythonanywhere.com](https://www.pythonanywhere.com) (free account)

2. **Upload your files**:
   - Go to "Files" tab
   - Create a new directory (e.g., `SOFI-2035-Info-Viz-code`)
   - Upload all files: `viz.py`, `data1.xlsx`, `requirements.txt`, `wsgi.py`

3. **Open a Bash console** and install dependencies:
```bash
cd SOFI-2035-Info-Viz-code
pip3.10 install --user -r requirements.txt
```

4. **Configure Web App**:
   - Go to "Web" tab
   - Click "Add a new web app"
   - Choose "Manual configuration"
   - Select Python 3.10
   - Set source code directory: `/home/YOUR_USERNAME/SOFI-2035-Info-Viz-code`
   - Edit WSGI configuration file:
     - Replace `YOUR_USERNAME` in `wsgi.py` with your actual username
     - Copy contents of `wsgi.py` to the WSGI configuration file

5. **Reload** your web app

6. **Access** your dashboard at: `https://YOUR_USERNAME.pythonanywhere.com` -->

## How to Use

### What-If Simulator
1. Select indicators from the dropdown
2. Enter percentage change values
3. Choose growth type (one-time, compound, or annual rate)
4. Click "Apply Changes" to see the impact on SOFI

### Historical Trends
- Toggle between normalized and original values
- Select multiple indicators to compare trends
- Observe the projection line at 2024 (last historical year)

### Correlations
- Use scatter plots to explore relationships between indicators
- Check the correlation bar chart to identify key SOFI drivers

### Indicator Analysis
- Select two years to compare
- View all indicators side-by-side
- Switch between normalized and original scales

## Data Coverage

The dashboard analyzes **29 indicators** across multiple dimensions:

**Economic**: GNI per capita, Income Inequality, Unemployment, Poverty, FDI, R&D

**Social**: Life expectancy, Mortality rate, Health expenditure, Literacy, School enrollment, Gender equality

**Environmental**: CO2 emissions, Renewable energy, Forest area, Biocapacity, Renewable freshwater

**Governance**: CPIA, Freedom Rights, Wars, Terrorism, Refugees

**Technology**: Patents, Internet Users

## Key Technologies

- **[Dash](https://dash.plotly.com/)**: Interactive web framework for Python
- **[Plotly](https://plotly.com/)**: Advanced graphing library
- **[Pandas](https://pandas.pydata.org/)**: Data manipulation and analysis
- **[OpenPyXL](https://openpyxl.readthedocs.io/)**: Excel file handling

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/SOFI-2035-Info-Viz-code.git
cd SOFI-2035-Info-Viz-code
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python viz.py
```

5. **Open your browser**
Navigate to `http://127.0.0.1:8050/`
