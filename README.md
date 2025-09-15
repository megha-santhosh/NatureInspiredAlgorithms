# 🧠 Stress & Sleep Analysis using GA + ACO + ML  

This project explores **feature selection and hyperparameter optimization** on two datasets:  

- **StressLevelDataset.csv**  
- **Sleep_health_and_lifestyle_dataset.csv**  

We use a combination of **Genetic Algorithm (GA)** and **Ant Colony Optimization (ACO)** with **Random Forest** and **Decision Tree** models to improve prediction accuracy.  

---

## 🚀 Features  

1. **Genetic Algorithm (GA) + Random Forest**  
   - GA is applied for **feature selection**.  
   - Random Forest is trained on both **full features** and **GA-selected features**.  
   - R² score is compared before and after GA optimization.  

2. **Ant Colony Optimization (ACO) + Decision Tree**  
   - ACO is used for **hyperparameter tuning** of Decision Trees.  
   - Parameters optimized: `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`.  
   - Accuracy is compared before and after ACO optimization.  

3. **Datasets Used**  
   - `StressLevelDataset.csv`: Predicts stress levels using physiological and lifestyle features.  
   - `Sleep_health_and_lifestyle_dataset.csv`: Includes sleep patterns, lifestyle habits, BMI, and stress levels.  

---

## 📂 Project Structure  

├── StressLevelDataset.csv
├── Sleep_health_and_lifestyle_dataset.csv
├── stress_sleep_analysis.py # Main script
├── requirements.txt # Dependencies
├── README.md # Documentation
