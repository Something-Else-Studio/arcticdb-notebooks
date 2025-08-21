# ArcticDB Sports Analytics Notebooks

This repository contains Jupyter notebooks demonstrating sports analytics using ArcticDB, specifically implementing the Pythagorean won-loss formula across multiple sports leagues.

## 📊 What's Inside

The `pythagorean-won-loss-formula/` directory contains:

- **`prepare.ipynb`** - Data preparation notebook that loads sports data and sets up ArcticDB storage [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Something-Else-Studio/arcticdb-notebooks/blob/main/pythagorean-won-loss-formula/prepare.ipynb)
- **`sports_from_arctic.ipynb`** - Main analysis notebook with statistical modeling and visualizations [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Something-Else-Studio/arcticdb-notebooks/blob/main/pythagorean-won-loss-formula/sports_from_arctic.ipynb)
- **`sports.csv`** - Dataset containing game-level data across multiple sports leagues

## 🏆 Key Features

- **ArcticDB Integration**: Efficient storage and retrieval of sports data using ArcticDB's high-performance database
- **Multi-League Analysis**: Analysis across 9 different sports leagues:
  - AFL (Australian Football League)
  - NBA (National Basketball Association)
  - NFL (National Football League)
  - MLB (Major League Baseball)
  - NHL (National Hockey League)
  - EPL (English Premier League)
  - IPL (Indian Premier League)
  - LAX (Lacrosse)
  - SUP (Super Rugby)
- **Pythagorean Won-Loss Formula**: Implementation of Bill James' formula to predict team performance
- **Statistical Modeling**: OLS regression analysis to estimate lambda coefficients for each league
- **Data Visualization**: Comprehensive charts and plots using seaborn and matplotlib

## 📋 Requirements

```bash
pip install arcticdb pandas numpy statsmodels seaborn matplotlib
```

Main dependencies:

- `arcticdb` - High-performance database for time series data
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `statsmodels` - Statistical modeling
- `seaborn` - Statistical data visualization
- `matplotlib` - Plotting library

## 🚀 Usage

1. **Data Preparation**:

   ```bash
   jupyter notebook pythagorean-won-loss-formula/prepare.ipynb
   ```

   This notebook:

   - Loads the sports.csv dataset
   - Cleans and preprocesses the data
   - Sets up ArcticDB storage with league-specific symbols
   - Stores data for efficient retrieval

2. **Main Analysis**:
   ```bash
   jupyter notebook pythagorean-won-loss-formula/sports_from_arctic.ipynb
   ```
   This notebook:
   - Connects to ArcticDB and retrieves sports data
   - Performs statistical analysis of points scored across leagues
   - Implements the Pythagorean won-loss formula
   - Generates visualizations and regression results

## 📈 Data

The dataset contains game-level data with the following key fields:

- `YEAR_ID` - Season year
- `TEAM_ID` - Team identifier
- `PS` (Points Scored) - Points scored by the team
- `PA` (Points Allowed) - Points allowed by the team
- `wi` - Win indicator (1 for win, 0 for loss)
- `LEAGUE` - Sport league identifier

**Analysis Period**: Primary focus on 2010-2019 seasons for consistency across leagues.

## 🔍 Key Results

The analysis reveals significant differences in competitive balance across sports:

- **Information Ratio (IR)**: Measures scoring consistency (Mean/Standard Deviation)

  - Highest IR: NBA (8.08) - most predictable scoring
  - Lowest IR: EPL (1.09) - most variable scoring

- **Lambda Coefficients**: Measure how strongly point differential predicts wins
  - Highest λ: NBA (14.32) - point differential strongly predicts outcomes
  - Lowest λ: EPL (1.24) - point differential weakly predicts outcomes

## ⚡ About ArcticDB

[ArcticDB](https://github.com/man-group/ArcticDB) is a high-performance database designed for time series data. In this project, it provides:

- Efficient storage of large sports datasets
- Fast querying and aggregation capabilities
- Symbol-based organization (one symbol per league)
- Seamless integration with pandas DataFrames

Perfect for sports analytics where you need to quickly access and analyze historical performance data across multiple dimensions (teams, seasons, leagues).

## 📝 License

This project is part of the ArcticDB examples and follows the same licensing terms.
