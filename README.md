# 🩺 Diabetes Prediction System

A Machine Learning-based web application that predicts the likelihood of diabetes based on patient health information. The application uses a trained **Multi-Layer Perceptron (MLP)** model and provides predictions through an interactive **Streamlit web interface**.

---

## 🚀 Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib

---

## ✨ Features

- User-friendly web interface
- Real-time diabetes prediction
- Machine Learning-based prediction using an MLP model
- Patient health data input
- Feature preprocessing using a trained scaler
- Interactive Streamlit application

---

## 📁 Project Structure

```text
Diabetes_Prediction/
│
├── app.py
├── diabetes_mlp.keras
├── diabetes_scaler.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

### File Description

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit web application |
| `diabetes_mlp.keras` | Trained MLP Machine Learning model |
| `diabetes_scaler.pkl` | Saved feature scaler used for preprocessing |
| `requirements.txt` | List of required Python libraries |
| `.gitignore` | Specifies files and folders ignored by Git |
| `README.md` | Project documentation |

---

## 🔄 Project Workflow

```text
User Input
    ↓
Streamlit Web Interface
    ↓
Input Data Collection
    ↓
Feature Scaling
(diabetes_scaler.pkl)
    ↓
Trained MLP Model
(diabetes_mlp.keras)
    ↓
Diabetes Prediction
    ↓
Result Displayed to User
```

### Workflow Explanation

1. **User Input**  
   The user enters patient health information through the Streamlit web application.

2. **Data Collection**  
   The application collects the input values required for diabetes prediction.

3. **Feature Scaling**  
   The input data is transformed using the saved scaler (`diabetes_scaler.pkl`) to ensure that the features are properly scaled.

4. **Model Prediction**  
   The processed input data is passed to the trained MLP model (`diabetes_mlp.keras`).

5. **Prediction Result**  
   The model predicts the likelihood of diabetes.

6. **Result Display**  
   The prediction result is displayed to the user through the Streamlit interface.

---

## 🖥️ Application Flow

```text
Patient Details
      ↓
Enter Health Parameters
      ↓
Click Predict
      ↓
Data Preprocessing
      ↓
MLP Model Prediction
      ↓
Display Result
```

---

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/TPS1234795/Diabetes_Prediction.git
```

### 2. Navigate to the Project Folder

```bash
cd Diabetes_Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

### 5. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will run locally at:

```text
http://localhost:8501
```

---

## 🧠 Machine Learning Model

The project uses a **Multi-Layer Perceptron (MLP)** neural network for diabetes prediction.

The model processes patient health parameters and generates a prediction based on patterns learned from the training dataset.

---

## 📌 Future Improvements

- Improve model accuracy
- Add more health parameters
- Deploy the application online
- Add user authentication
- Store prediction history
- Improve UI and visualization

---

## 👩‍💻 Author

**Taniprava Sahoo**

B.Tech Computer Science Engineering (AI & ML)
