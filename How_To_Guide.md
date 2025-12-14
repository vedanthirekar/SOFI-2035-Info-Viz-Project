# SOFI Dashboard - How-To Guide

## Quick Access

**Live Dashboard:** https://sofi2035.pythonanywhere.com/

**GitHub Repository:** https://github.com/vedanthirekar/SOFI-2035-Info-Viz-Project

---

## Setup and Installation

### Prerequisites
- Python 3.10+
- pip package manager

### Installation Steps

```bash
# 1. Clone repository
git clone [your-github-url]
cd SOFI-2035-Info-Viz-code

# 2. Create & activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run application
python viz.py
```

Open browser: `http://127.0.0.1:8050`

---

## How to Navigate the Dashboard

### Tab 1: Historical Trends

| View | Description |
|------|-------------|
| SOFI Index Over Time | Line chart of overall index (1990-2035) |
| Year-over-Year Change | Bar chart of percentage changes |
| Individual Indicator Trends | Multi-line comparison of selected indicators |

**Interactions:**
- Select indicators from dropdown to compare
- Toggle Normalized/Original values
- Hover for detailed data points

---

### Tab 2: What-If Simulator

**Step-by-step:**
1. Select indicator → Click "Add Indicator"
2. Enter % change (positive/negative)
3. Choose growth type:
   - *One-time:* Permanent shift to predicted curve
   - *Compound:* Year-over-year exponential from 2024
   - *Annual rate:* Linear growth from 2024
4. Click "Apply Changes"
5. "Clear All" to reset

**Reading Results:**
- Dashed line = Baseline
- Solid line = Adjusted
- SOFI panel shows combined impact

---

### Tab 3: Correlations

| View | Description |
|------|-------------|
| Scatter Plot | Compare two indicators (colored by year) |

**Interactions:**
- Select X/Y indicators from dropdowns
- Hover for year and values
- Axis labels show measurement scales

---

### Tab 4: Indicator Analysis

**How to use:**
1. Select two years to compare
2. Filter by category (Economic, Social, Environmental, Governance, Technology)
3. Toggle Normalized/Original view

**Interactions:**
- Hover bars for indicator scale and values
- Log scale used for original values

---

## Key Features Summary

- **29 Indicators** across 5 dimensions
- **1990-2035** historical + projected data
- **3 Growth Models** for scenario simulation
- **Category Filters** for focused analysis
- **Dual Views** (normalized & original values)
- **Interactive Tooltips** with measurement scales
