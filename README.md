# Movie Genre Classification using Machine Learning

Automated movie genre classification system comparing three supervised machine learning algorithms: Decision Tree, Naive Bayes, and Random Forest.

## Project Overview

This project implements and evaluates three machine learning algorithms for automated classification of movies into distinct genres (Animation, Horror, Documentary) using metadata features and textual descriptions. The system addresses scalability challenges faced by streaming platforms in manual content categorization.

### Key Achievements
- **Best Performance:** Random Forest with **81.72% accuracy**
- **Baseline Comparison:** Decision Tree (72.58%) and Naive Bayes (71.24%)
- **Significant Improvement:** 10.48% accuracy gain over Naive Bayes through ensemble learning
- **Production-Ready:** Exceeds typical text classification benchmarks (60-75%)

##Algorithms Implemented

| Algorithm | Notebook File | Accuracy | Precision | Recall | F1-Score |
|-----------|--------------|----------|-----------|--------|----------|
| **Random Forest** | `random_forest.ipynb` | **81.72%** | 0.8216 | 0.8172 | 0.8144 |
| Decision Tree | `decision_tree.ipynb` | 72.58% | 0.7265 | 0.7258 | 0.7260 |
| Naive Bayes | `naive_bayes.ipynb` | 71.24% | 0.7276 | 0.7124 | 0.7058 |

### Algorithm Highlights

**Random Forest Classifier**
- Ensemble of 500 decision trees with majority voting
- Implements bootstrap sampling and feature randomization
- Best performance with balanced precision-recall trade-off
- Ideal for production deployment

**Decision Tree Classifier**
- Single tree with max_depth=50 and Gini criterion
- Fast training and prediction
- Interpretable decision rules but prone to overfitting

**Naive Bayes Classifier**  
- Probabilistic approach with MultinomialNB
- Fastest training time with minimal computational requirements
- Limited by feature independence assumption

##Dataset Specifications

- **Source:** Hugging Face repository (Pablinho/movies-dataset)
- **Total Movies:** 1,856 films (post-filtering)
- **Genre Distribution:**
  - Animation: 804 movies
  - Horror: 868 movies
  - Documentary: 184 movies
- **Train-Test Split:** 80-20 stratified sampling
  - Training: 1,484 samples
  - Testing: 372 samples

### Strategic Genre Selection

Initial experiments with Drama, Action, and Comedy achieved only 35% accuracy due to overlapping characteristics. Switching to three highly distinct genres (Animation, Horror, Documentary) improved accuracy by over 45 percentage points.

## 🔧 Feature Engineering

### Numerical Features (7 total)
- **Original Features:** Popularity, Vote_Count, Vote_Average
- **Engineered Features:**
  - `Language_Encoded`: Numerical encoding of original language
  - `Release_Year`: Temporal feature extracted from release date
  - `Vote_Score`: Vote_Count × Vote_Average (engagement metric)
  - `Popularity_Per_Vote`: Popularity ÷ (Vote_Count + 1) (normalized popularity)

### Text Features (100 total)
- **Method:** TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- **Source:** Movie plot descriptions and overviews
- **Configuration:**
  - Maximum 100 features
  - English stop words removed
  - Bigram support (1-2 word combinations)
  - Captures genre-specific keywords (e.g., "animated", "terrifying", "documentary")

**Total Feature Vector:** 107 features per movie

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.8+** - Primary programming language
- **Jupyter Notebook** - Interactive development environment

### Libraries
- **pandas** - DataFrame operations, data cleaning, feature engineering
- **numpy** - Numerical computations and array operations
- **scikit-learn** - Machine learning framework
  - `DecisionTreeClassifier` - Decision Tree implementation
  - `MultinomialNB` - Naive Bayes implementation
  - `RandomForestClassifier` - Random Forest implementation
  - `TfidfVectorizer` - Text feature extraction
  - `train_test_split` - Stratified data splitting
  - `StandardScaler` - Feature normalization
- **matplotlib** - Data visualization and performance charts
- **seaborn** - Statistical visualizations and confusion matrices
- **scipy** - Sparse matrix operations for efficient feature combination

##Installation & Setup

### Prerequisites
```bash
