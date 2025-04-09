# 🩺 Vital Disease Prediction & Diet Recommendation System

An AI-powered, interactive desktop application that predicts potential diseases from patient vital signs and recommends tailored diet plans — empowering users with early healthcare insights and preventive guidance.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg) ![ML](https://img.shields.io/badge/MachineLearning-RandomForest-orange.svg) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

---

## 🌟 What Makes This Project Unique?

✅ **Multi-Disease Prediction (Multi-label Classification):**  
Unlike traditional systems that detect one disease at a time, this app predicts multiple possible conditions per patient, giving a **more comprehensive diagnosis**.

✅ **Real-Time Diet Recommendations:**  
Once diseases are predicted, the system **automatically suggests personalized diet plans**, making it a complete health management assistant.

✅ **End-to-End Architecture:**  
From dataset preprocessing, model training, and evaluation to a full-featured GUI — this system is **fully self-contained and deployable**.

✅ **Interactive Data Visualization:**  
Dashboards visualize disease trends, gender/age distributions, and vital metrics correlation — **useful for public health researchers and analysts**.

✅ **Desktop-Friendly Application:**  
Built with **Tkinter**, this app is lightweight, fast, and easy to use on any machine without browser dependencies.

---

## 🧠 Tech Stack & Tools

| Layer       | Technologies                                 |
|-------------|----------------------------------------------|
| Language    | Python 3.8+                                  |
| Machine Learning | Scikit-learn, Pandas, NumPy                  |
| Visualization | Matplotlib, Seaborn                           |
| GUI         | Tkinter                                       |
| Model       | MultiOutput Random Forest Classifier         |
| Data        | Cleaned CSV + Pretrained Pickle Models       |
| Packaging   | Joblib, Pickle, StandardScaler, LabelBinarizer |

---

## 🧪 Features

### 📍 Health Prediction
- Inputs: Hemoglobin, BP, Heart Rate, LDL, HbA1c, CRP, Vitamin D, etc.
- Output: Probable diseases (e.g., Anemia, Diabetes, Hypertension)

### 🥗 Diet Plan Generator
- Auto-suggests diets based on predicted diseases  
- Custom logic tailored to common lifestyle ailments

### 📊 Analytics Dashboard
- Age-wise and gender-wise disease trends  
- Vital stats distribution & correlation heatmaps  
- Top 10 most diagnosed conditions

### 🔍 Patient Management
- Add new patients, update records, filter by condition  
- Searchable database with exportable insights

### 🖼️ Clean UI/UX
- Navigation bar for switching between prediction, data analysis, and reports  
- Dropdowns, tables, and real-time updates

---

## 📁 Project Structure

```bash
📦 vital-disease-predictor/
├── model.py               # ML model training and evaluation
├── interface.py           # Tkinter GUI app logic
├── analysis.py            # EDA and dataset summary
├── vital_disease_prediction_dataset.csv
├── vital_disease_predictor.pkl
├── label_binarizer.pkl
├── scaler.pkl
└── README.md
```

---

## 🚀 How to Run

### ✅ Prerequisites
- Python 3.8+
- Install dependencies:
```bash
pip install pandas scikit-learn matplotlib seaborn joblib
```

### 🧠 Model Training (Optional)
```bash
python model.py
```

### 🖥 Launch the App
```bash
python interface.py
```

---

## 📈 Results Summary

- ✅ **Model Accuracy:** ~89%  
- ✅ **Hamming Loss:** Low, ensuring multi-label accuracy  
- ✅ **Top Features:** Hemoglobin, LDL, BP, Heart Rate  

> 📊 The model's feature importance identifies **which vitals influence diseases most**, aiding interpretability for medical professionals.

---

## 🔐 Security & Data Handling
- Environment-safe loading of models  
- Robust input validation  
- Modular & clean code structure for future API or cloud upgrades  

---

## 🔮 Future Scope

- 🌐 Web deployment (Flask/Django)  
- 📱 Mobile app with camera-based OCR inputs  
- 🤖 Integration with IoT medical wearables  
- 🌍 Multilingual UI for inclusivity  

---

## 🙋‍♀️ About the Author

**👩🏻 Suchita Yerramsetty**  
- 🔭 Data Science Enthusiast | ML Developer  
- 🌐 [LinkedIn](https://www.linkedin.com/in/yerramsetty-sai-venkata-suchita-suchi1234/)  
- 💻 [GitHub](https://github.com/yerramsettysuchita)  
- 📧 suchitayerramsetty999@gmail.com

---

## 📄 License

This project is licensed under the **MIT License** – use it, learn from it, and expand it!

---

> “Preventive healthcare is the future. This project brings that future a little closer.”
