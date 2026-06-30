# ChurnAI — Customer Churn Prediction Engine 📊🤖

ChurnAI is a real-time analytics dashboard powered by an **Artificial Neural Network (ANN)** that predicts customer churn probability based on historical bank customer demographics and financial behavior.

The application is built using a deep learning model developed in **TensorFlow/Keras** and presented through a custom dark-themed **Streamlit** fintech interface.

---

## 🌟 Key Features

*   **🧠 Deep Learning Engine:** Uses a trained Artificial Neural Network (ANN) model to output precise churn probability metrics.
*   **🎨 Premium Fintech Design:** Rebuilt with a custom dark-mode interface featuring animated floating blobs, glassmorphic profile cards, and clean visual layouts.
*   **📊 Dynamic Risk Indicator:** Visually scales churn danger through colored risk progress bars (green/safe to red/danger) based on model confidence.
*   **📋 Detailed Profile Breakdown:** Outputs a structured customer profile summary alongside model outputs for instant verification.

---

## 🏗️ Project Structure

```
annclassification/
├── app.py                      # Streamlit application with custom CSS
├── requirements.txt            # Project dependencies
├── Churn_Modelling.csv         # Bank customer churn dataset
├── model.h5                    # Trained Keras ANN model weights
├── scaler.pkl                  # Fitted StandardScaler object
├── label_encoder_gender.pkl    # Fitted LabelEncoder for Gender
├── onehot_encoder_geo.pkl      # Fitted OneHotEncoder for Geography
└── experiments.ipynb           # Model development and testing notebook
```

---

## 🚀 Getting Started

### 📋 Prerequisites

To run this project locally, set up a virtual environment and install the required dependencies:

```bash
# 1. Create a virtual environment
python3 -m venv venv

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Install packages
pip install -r requirements.txt
```

### 💻 Running the Application

Launch the Streamlit server locally:
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your web browser.

---

## 📄 License
This project is licensed under the MIT License.
