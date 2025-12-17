Breast Cancer Detection Using Machine Learning

This project predicts whether a breast tumor is Cancerous (Malignant) or Non-Cancerous (Benign) using Machine Learning based on 5 clinical features from the Wisconsin Breast Cancer Dataset.

The application is built using Python, Scikit-learn, and Flask, with a simple and user-friendly web interface.

📌 Features

Predicts Cancer / No Cancer

Uses Logistic Regression

Uses StandardScaler for feature normalization

Web-based interface using Flask

Reset button for multiple predictions

Clear explanation of Class 0 / Class 1

🧠 Machine Learning Details

Algorithm: Logistic Regression

Input Features (5):

Radius Mean

Texture Mean

Perimeter Mean

Area Mean

Smoothness Mean

Output:

0 → Benign (No Cancer)

1 → Malignant (Cancer Detected)

📂 Project Structure
breast cancer detection/
│
├── app.py                     # Flask application
├── train_model.py             # Model training script
├── data.csv                   # Dataset
├── breast_cancer_model.pkl    # Trained ML model
├── scaler.pkl                 # Feature scaler
│
├── templates/
│   └── index.html             # Frontend UI
│
├── static/
│   └── image7.jpg             # Sample image
│
└── README.md                  # Project documentation

⚙️ Requirements

Make sure Python 3.11 is installed.

Required Libraries

Flask

NumPy

Pandas

Scikit-learn

Joblib

🔧 Installation

Open PowerShell / CMD in the project folder and run:

py -3.11 -m pip install flask numpy pandas scikit-learn joblib

🏗️ Step 1: Train the Model

This step creates:

breast_cancer_model.pkl

scaler.pkl

Run:

py -3.11 train_model.py


✅ Output:

breast_cancer_model.pkl and scaler.pkl created successfully

🚀 Step 2: Run the Flask Application

Start the web application using:

py -3.11 app.py


You should see:

Running on http://127.0.0.1:5000

🌐 Step 3: Open in Browser

Open any browser and go to:

http://127.0.0.1:5000

🧪 How to Use the Application

Enter values for all 5 features

Click Predict

View result:

Malignant (Cancer Detected)

Benign (No Cancer)

Click Reset to clear inputs

📊 Dataset Information

Dataset: Wisconsin Breast Cancer Dataset

Source: Kaggle / UCI ML Repository

Data Type: Tabular (CSV)

🎓 Viva / Interview One-Liner

“This project uses a machine learning model trained on clinical features to predict breast cancer and deploys it as a web application using Flask.”

🔮 Future Enhancements

Add confidence score (%)

Add image-based cancer detection (CNN)

Deploy on cloud (Render / AWS)

Generate downloadable prediction reports
