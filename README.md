# 🎾 Tennis Match Outcome Prediction System

A comprehensive machine learning system for predicting professional tennis match outcomes using 25 years of ATP tournament data (2000-2024).


## 📊 Project Overview

This project implements multiple machine learning algorithms to predict the outcomes of professional tennis matches using historical ATP data. The system processes 74,906+ matches spanning 25 years and achieves **65.7% prediction accuracy** using optimized ensemble methods.

### 🎯 Key Achievements
- **65.7% prediction accuracy** with Random Forest ensemble
- **74,906+ matches** analyzed from ATP tournaments (2000-2024)
- **Zero data leakage** with rigorous temporal validation
- **40+ engineered features** capturing tennis-specific patterns
- **Multiple ML algorithms** compared and optimized

## 🏗️ Project Structure

```
tennis_model/
├── data/                          # ATP dataset directory (not included)
│   ├── atp_matches_*.csv         # Match data by year
│   ├── atp_players.csv           # Player information
│   └── atp_rankings_*.csv        # Historical rankings
├── fast_tennis_processor.py      # Core data processing pipeline
├── decision_rank.py              # Rankings-only decision tree
├── decision_tree.py              # Full decision tree model
├── random_forest_tennis.py       # Random Forest implementation (65.7% accuracy)
├── xgBoost.py                    # XGBoost with hyperparameter tuning
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 8GB+ RAM (for processing large datasets)
- ATP tennis dataset (see Data Requirements below)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/AshutoshRegmi/TennisModel.git
cd TennisModel
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up data directory**
```bash
mkdir data
# Place ATP CSV files in the data/ directory
```

### 📁 Data Requirements

This project requires ATP tennis datasets:
- `atp_matches_YYYY.csv` (match data by year, 2000-2024)
- `atp_players.csv` (player information)
- `atp_rankings_*.csv` (historical rankings)

**Data Source**: [Jeff Sackmann's Tennis Datasets](https://github.com/JeffSackmann/tennis_atp)

## 🎯 Usage

### Run Individual Models

**1. Rankings-Only Baseline**
```bash
python decision_rank.py
```
Simple decision tree using only player rankings.

**2. Decision Tree (All Features)**
```bash
python decision_tree.py
```
Full decision tree with all engineered features.

**3. Random Forest (Best Performance)**
```bash
python random_forest_tennis.py
```
Optimized Random Forest achieving 65.7% accuracy.

**4. XGBoost with Hyperparameter Tuning**
```bash
python xgBoost.py
```
Advanced gradient boosting with feature selection and ensemble methods.

### Custom Analysis

```python
from fast_tennis_processor import FastTennisProcessor

# Load and process data
processor = FastTennisProcessor(data_path='./data/')
X, y, df = processor.run_fast_pipeline(start_year=2000, end_year=2024)

# Your custom ML pipeline here
```

## 📈 Model Performance

| Model | Accuracy | Features | Notes |
|-------|----------|----------|-------|
| **Random Forest** | **65.7%** | 40+ engineered | Best overall performance |
| XGBoost Ensemble | 64.8% | 25 selected | Hyperparameter optimized |
| Decision Tree | 65.1% | All features | Strong baseline |
| Rankings Only | 62.3% | 3 ranking features | Minimal baseline |

### 🎯 Performance Breakdown
- **Ranking features**: 70%+ importance (player1_rank, player2_rank, rank_difference)
- **Elite indicators**: Top 10/20/50 classifications significantly improve predictions
- **Tournament context**: Grand Slams vs regular tournaments matter
- **Age factors**: Prime age (24-29) vs veteran (30+) patterns

## 🔧 Technical Implementation

### Feature Engineering

**Core Features (16 base features):**
- Player rankings and rank differences
- Player ages and age gaps
- Tournament level and round
- Surface type (Hard, Clay, Grass)
- Player handedness

**Advanced Features (40+ total):**
- Non-linear ranking transformations (`rank_ratio`, `rank_log_diff`)
- Elite tier classifications (Top 5, 10, 20, 50)
- Skill gap indicators (`huge_gap`, `elite_vs_regular`)
- Tournament importance weights
- Historical form indicators (where applicable)

### Data Integrity

**Preventing Data Leakage:**
- Random player1/player2 assignment (no winner/loser bias)
- Temporal validation splits
- No future information in features
- Rigorous train/test separation

**Validation Strategy:**
- 80/20 train/test split
- 5-fold cross-validation
- Stratified sampling for balanced classes

## 📊 Key Insights

### 🏆 What Drives Tennis Predictions?

1. **Rankings are King**: 70%+ of predictive power comes from ATP rankings
2. **Non-linear Skill Gaps**: Difference between rank 1-10 >> rank 100-110
3. **Elite Tiers Matter**: Top 10 vs Top 50 vs unranked have distinct patterns
4. **Context Counts**: Grand Slams have different dynamics than regular tournaments
5. **Age Factors**: Prime age players (24-29) have advantage over veterans

### 🎾 Tennis-Specific Patterns Discovered

- **Upset Potential**: Players ranked 50+ beating Top 20 has identifiable patterns
- **Surface Effects**: Clay vs Hard vs Grass create different prediction dynamics
- **Tournament Pressure**: Performance varies by round (early vs semifinals/finals)
- **Elite Clashes**: Top 10 vs Top 10 matches are harder to predict (more random)

## 🔬 Advanced Features

### Model Optimization
- **GridSearchCV**: Exhaustive hyperparameter search
- **Feature Selection**: SelectKBest with F-classification scores
- **Ensemble Methods**: Weighted voting classifiers
- **Cross-Validation**: Multiple validation strategies

### Performance Monitoring
- **Confusion Matrix Analysis**: Detailed prediction breakdowns
- **Feature Importance**: Understanding model decisions
- **Sample Predictions**: Real match prediction examples with confidence scores

## 📋 Requirements

```txt
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
xgboost>=1.7.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
jupyter>=1.0.0
tqdm>=4.64.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

### Improvement Ideas
- [ ] Add serve statistics (aces, double faults, break points)
- [ ] Implement head-to-head historical records
- [ ] Add surface-specific player performance
- [ ] Neural network implementations
- [ ] Real-time prediction API
- [ ] Betting odds integration


## 🙏 Acknowledgments

- **Data Source**: Tennis Abstract and Jeff Sackmann's comprehensive ATP datasets
- **Inspiration**: Professional tennis analytics and sports betting research
- **Libraries**: Scikit-learn, XGBoost, Pandas ecosystem


⭐ **Star this repository if it helped you!**

## 🔍 Future Work

### Potential Improvements
- **Match Statistics Integration**: Serve %, aces, break points from match data
- **ELO Rating System**: Dynamic player strength ratings
- **Surface Specialization**: Player-specific surface performance history
- **Injury/Fitness Data**: Player condition indicators
- **Live Prediction API**: Real-time match outcome predictions
- **Deep Learning**: LSTM networks for sequence prediction

### Research Applications
- Sports analytics and performance modeling
- Betting market efficiency analysis
- Tournament seeding optimization
- Player development insights

---

**Built with ❤️ for tennis analytics**
