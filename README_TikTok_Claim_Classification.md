# 🧠 TikTok Claim Classification Project  

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-yellow?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange?logo=scikit-learn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green?logo=plotly)
![Status](https://img.shields.io/badge/Project_Status-Completed-brightgreen)

---

## 📘 **Table of Contents**
- [🎯 Objective](#-objective)
- [📂 Project Overview](#-project-overview)
- [🧹 Data Cleaning & Preprocessing](#-data-cleaning--preprocessing)
- [🔍 Exploratory Text Analysis](#-exploratory-text-analysis)
- [🤖 Model Building – Stacking Classifier](#-model-building--stacking-classifier)
- [⚙️ Feature Engineering](#️-feature-engineering)
- [🧩 Model Training & Evaluation](#-model-training--evaluation)
- [📊 Model Performance](#-model-performance)
- [🧰 Tech Stack](#-tech-stack)
- [📈 Visual Output](#-visual-output)
- [🧠 Key Learnings](#-key-learnings)
- [🚀 Future Improvements](#-future-improvements)
- [👨‍💻 Author](#-author)

---

## 🎯 **Objective**
To classify TikTok videos as **“Claim”** or **“Opinion”** based on their **video transcription text** and metadata.  
This helps analyze how factual statements versus personal opinions are distributed across social media content.

---

## 📂 **Project Overview**
**Dataset:** `tiktok_dataset.csv`  
**Records:** 19,382  
**Columns:** 10  

| Feature | Description |
|----------|-------------|
| claim_status | Target label (`claim` / `opinion`) |
| video_transcription_text | Text spoken in the video |
| video_duration_sec | Video length in seconds |
| verified_status | Whether creator is verified |
| author_ban_status | Account status |
| video_view_count | Total video views |
| video_like_count | Total likes |
| video_share_count | Total shares |
| video_download_count | Total downloads |
| video_comment_count | Total comments |

---

## 🧹 **Data Cleaning & Preprocessing**
1. Removed non-informative columns (`video_id`, unnamed columns).  
2. Handled missing values (`NaN`) using drop method.  
3. Applied **text cleaning and normalization**:
   - Converted to lowercase  
   - Removed punctuation & non-alphabetic characters  
   - Removed English stopwords (via NLTK)  
4. Tokenized text into individual words.  

```python
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return [w for w in text.split() if w not in stop_words]
```

---

## 🔍 **Exploratory Text Analysis**
Extracted most frequent words for both `claim` and `opinion` transcriptions:

| **Category** | **Frequent Words** |
|---------------|--------------------|
| Claim | media, learned, claim, read, friend, discovered |
| Opinion | world, opinion, friends, view, say, family |

These insights later helped generate keyword-based features used for classification.

---

## 🤖 **Model Building – Stacking Classifier**

A **Stacking Ensemble Model** was implemented using Scikit-learn.

**Base Learners:**
- Logistic Regression  
- Decision Tree Classifier  
- Support Vector Classifier (SVC)

**Meta Learner:**
- Random Forest Classifier  

> 💡 Stacking combines multiple models to leverage their individual strengths and minimize bias/variance.

---

## ⚙️ **Feature Engineering**
- One-hot encoding for categorical features (`verified_status`, `author_ban_status`, `label`).  
- Standardized numeric variables (`video_like_count`, `video_view_count`, etc.) using `StandardScaler`.  
- Derived feature: `video_duration_minutes = video_duration_sec / 60`

---

## 🧩 **Model Training & Evaluation**

**Train-Test Split:** 67% Train / 33% Test  
**Evaluation Metrics:** Accuracy, F1-score, Precision, Recall, Cross-validation  

```python
base_learners = [
    ('lr', LogisticRegression(max_iter=2300)),
    ('dt', DecisionTreeClassifier()),
    ('svm', SVC(probability=True))
]
meta_model = RandomForestClassifier(n_estimators=300)
stacking = StackingClassifier(estimators=base_learners, final_estimator=meta_model)
stacking.fit(X_train, y_train)
```

---

## 📊 **Model Performance**

| Metric | Score |
|--------|-------|
| **Accuracy** | 99.78% |
| **Precision** | 1.00 |
| **Recall** | 1.00 |
| **F1-Score** | 1.00 |
| **Cross-Validation Accuracy** | 99.66% ± 0.34% |

✅ **Interpretation:**  
The stacking model demonstrates exceptional predictive power with balanced performance across both classes.

---

## 📈 **Visual Output**

Confusion Matrix Visualization:
```python
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Claim', 'Opinion'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()
```

---

## 🧰 **Tech Stack**

| Category | Tools Used |
|-----------|-------------|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| NLP | NLTK |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib |
| Model | Stacking Ensemble (LR + DT + SVM + RF) |

---

## 🧠 **Key Learnings**
- Applied **NLP preprocessing** and **word frequency analysis**.  
- Understood **ensemble learning (stacking)** and its practical benefits.  
- Enhanced skills in **feature scaling**, **encoding**, and **model evaluation**.  
- Achieved strong results through **data-driven feature design**.

---

## 🚀 **Future Improvements**
- Integrate **TF-IDF** or **Word Embeddings (BERT)** for contextual understanding.  
- Develop a **Streamlit dashboard** for interactive claim detection.  
- Explore **deep learning** models (e.g., LSTM, Transformer-based classifiers).  
- Add **language support for non-English content**.

---

## 👨‍💻 **Author**
**Amritanshu Sharma**  
🎓 Data Analyst | 💡 Machine Learning | NLP

📍 Mumbai, India  
📧 [amritanshusharma16451@gmail.com]  
💼 [[Amritanshu](https://www.linkedin.com/in/amritanshu-51b746384/)]  
🐍 *“Turning data into decisions, one model at a time.”*