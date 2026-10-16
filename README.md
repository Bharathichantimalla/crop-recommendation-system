# 🌱 AI/ML Crop Recommendation System

## 📌 Project Overview
The Crop Recommendation System is a machine learning web application that recommends a suitable crop based on soil nutrients and environmental conditions. It uses a Random Forest Classifier and a simple interactive interface built with Streamlit.

## 🎯 Objectives
- Recommend suitable crops based on soil and weather data.
- Use machine learning to predict the most appropriate crop.
- Provide a user-friendly web interface for entering input values.
- Demonstrate a practical application of AI/ML in agriculture.

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Random Forest Classifier

## 📊 Input Features
The application uses the following inputs:
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

## ⚙️ How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/Bharathichantimalla/crop-recommendation-system.git
cd crop-recommendation-system
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

Activate it on Windows:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
python -m streamlit run app.py
```

The application will open in your browser, usually at `http://localhost:8501`.

## 📁 Project Structure
```text
crop-recommendation-system/
├── app.py
├── crop.csv
├── requirements.txt
└── README.md
```

## 🌾 Expected Output
The application displays a recommended crop based on the soil nutrient values and environmental conditions entered by the user.

## 🚀 Future Enhancements
- Add crop yield prediction.
- Improve the user interface.
- Include fertilizer recommendations.
- Explore weather data integration.

## 👩‍💻 Author
Bharathi Chantimalla

## 📄 Disclaimer
Crop recommendations are model predictions based on the dataset and should be considered informational guidance, not a substitute for local agricultural expertise.