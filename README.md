#  Feature Selection and Classification for Dataset

This project aims to analyze a dataset by applying multiple **feature selection techniques** and comparing their performance in predicting customer subscription behavior. The system includes both **a Python backend** and an interactive **Streamlit application** for easy data analysis and visualization.

---

##  Project Features

 Supports any CSV dataset upload  
 Choose the target (label) column from the data  
 Automatically encodes categorical features  
 Compares multiple feature selection algorithms:
- **SelectKBest**
- **PCA (Principal Component Analysis)**
- **Genetic Algorithm Feature Selection (GA)**

 Shows best-selected features  
 Saves results in `results.joblib`  
 Built-in classifier: Logistic Regression  
 Faster GA evaluation by sampling only first 300 rows  
 Automatically limits GA-selected features to max **50**

---

##  Project Structure

```
project/
│
├── src/
│   ├── utils.py            # Preprocessing & evaluation functions
│   ├── train_experiments.py
│   ├── baseline.py
│   ├── ga_selector.py      # Genetic Algorithm Feature Selector
│
├── web/
│   └── app_streamlit.py    # Streamlit UI for CSV upload & modeling
│
├── data/
│   └── 
│
└── results.joblib          # Saved results file
```

---

##  Methods Used

| Method | Description | Output |
|--------|-------------|--------|
| SelectKBest | Selects top features using statistical scoring | Top 10 features |
| PCA | Reduces dimensionality with principal components | 10 components |
| GA | Evolves best feature combination | Best ≤ 50 features |

---

##  Performance Metrics

-  Accuracy
-  F1 Score
-  Recall
-  Precision

Results are compared in the Streamlit UI.

---

##  How to Run the App

Click on this link to open the app:
https://dataanalysis-nedviemtv4tpvd8q9hzyj3.streamlit.app/

Then:

1️. Upload your CSV  
2️. Choose target column  
3️. Click **Start Analysis **  
4️. Check results & selected features

---

##  Main Libraries

```
streamlit
pandas
numpy
scikit-learn
joblib
```

---

##  Conclusion

This system proves the effectiveness of **feature selection** in improving classification accuracy.  
The **Genetic Algorithm** generally achieved the **highest score** on the tested dataset.

---
