# 🤖 AutoML Platform 🚀

An end-to-end **Automated Machine Learning (AutoML) system** that simplifies the entire ML pipeline — from data ingestion to model deployment.
The platform automatically trains multiple models, compares performance, and provides insights through reports and an interactive **Streamlit UI**.

---

## ✨ Features

* 📥 Automatic data ingestion and validation
* 🧹 Data preprocessing (missing values, encoding, scaling)
* 🔀 Train-test splitting
* 🧠 Feature engineering (polynomial features + selection)
* 🤖 Multi-model training (classification & regression)
* ⚙️ Hyperparameter tuning
* 🏆 Model leaderboard with best model selection
* 📊 Evaluation metrics and bias detection
* 🔍 Model explainability (feature importance)
* 📝 Automated report generation
* 💾 Model saving and persistence
* 💬 Interactive Streamlit web app

---

## 🛠️ Tech Stack

* **Language:** Python
* **Frameworks:** Scikit-learn, Pandas, NumPy
* **ML Models:** Regression & Classification algorithms
* **Tuning:** GridSearch / RandomSearch
* **Visualization:** Matplotlib / Seaborn
* **Frontend:** Streamlit

---

## 📁 Project Structure

```id="auto1"
AutoML-project/
├── app.py
├── configs/
├── data/
│   └── raw/
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── pipeline.py
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_selection.py
│   ├── hyperparameter_tuning.py
│   ├── evaluation.py
│   ├── model_leaderboard.py
│   ├── report_engine.py
│   ├── eda.py
│   └── ...
└── README.md
```

---

## 📦 Installation

### 1️⃣ Clone the repository

```bash id="auto2"
git clone https://github.com/your-username/AutoML-project.git
cd AutoML-project
```

### 2️⃣ Create virtual environment

```bash id="auto3"
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Mac/Linux
```

### 3️⃣ Install dependencies

```bash id="auto4"
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### 🔹 Run pipeline from terminal

```bash id="auto5"
python -m src.pipeline
```

### 🔹 Run Streamlit app

```bash id="auto6"
streamlit run app.py
```

---

## 📊 Input Data

* Supported formats: `.csv`, `.xlsx`
* Place files inside:

```id="auto7"
data/raw/
```

or upload via Streamlit UI

👉 Select the **target column** before running pipeline

---

## 📤 Outputs

* 💾 Trained models → `models/`
* 📊 EDA plots → `reports/`
* 📝 Text report → `reports/model_report.txt`
* 🏆 Leaderboard → CLI + Streamlit

---

## 🧠 How It Works

```id="auto8"
Data → Preprocessing → Feature Engineering → Model Training → Tuning → Evaluation → Leaderboard → Reports
```

---

## 🔍 Core Modules

* `pipeline.py` → Orchestrates full workflow
* `data_ingestion.py` → Data loading & validation
* `preprocessing.py` → Cleaning & splitting
* `feature_engineering.py` → Feature generation & selection
* `model_selection.py` → Model training & scoring
* `hyperparameter_tuning.py` → Optimization
* `evaluation.py` → Metrics & performance
* `model_leaderboard.py` → Ranking models
* `report_engine.py` → Report generation
* `eda.py` → Data analysis & visualization

---

## ⚡ Key Highlights

* ✔ Fully automated ML pipeline
* ✔ Supports both regression & classification
* ✔ Compares multiple models automatically
* ✔ Provides explainability & insights
* ✔ User-friendly UI for non-technical users

---

## ⚠️ Notes

* Avoid **target leakage** during feature engineering
* Heavy models (XGBoost, LightGBM) may take longer on first run
* Ensure clean data for best performance

---

## 🚀 Future Improvements

* 📈 MLflow integration for experiment tracking
* 🌐 Deployment (Docker / Cloud)
* 🧠 LLM-based insights & recommendations
* ⚡ Auto feature selection improvements
* 📊 Advanced visualization dashboard

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork and submit a PR.

---

## 📜 License

MIT License

---

## 👨‍💻 Author

**Prathmesh Wavhal**
AI & Data Science Enthusiast 🚀

---

⭐ Star this repo if you find it useful!
