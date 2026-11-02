Flipkart Customer Satisfaction (CSAT) Score Prediction

A Machine Learning project to predict customer satisfaction scores (CSAT: 1–5) using customer service data. The project includes data preprocessing, exploratory data analysis, model development, evaluation, and a Streamlit web application deployed on Render.

Project Overview

Customer satisfaction is an important indicator of customer experience. This project uses historical customer service records to predict CSAT scores and analyze customer satisfaction patterns.

The dataset contains 85,907 customer service records with a highly imbalanced target variable.

The objective is to build a classification model that predicts CSAT scores and helps identify dissatisfied customers.

Key Features

- Exploratory Data Analysis (EDA) to understand customer satisfaction patterns.
- Data cleaning and preprocessing.
- Handling class imbalance using SMOTE.
- Training and evaluating multiple classification models.
- Comparing model performance against a Logistic Regression baseline.
- Streamlit web application for CSAT score prediction.
- Deployment on Render.

Dataset

The dataset contains customer service records with the target variable:

CSAT Score: Customer satisfaction rating from 1 to 5.

The target distribution is imbalanced, with the majority of records belonging to CSAT score 5.

Some dataset features contain substantial missing values. Data preprocessing and feature selection are performed before model training.

«Dataset source and availability may depend on the original dataset license and access permissions.»

Tech Stack

Category| Tools
Programming Language| Python
Data Analysis| Pandas, NumPy
Data Visualization| Matplotlib, Seaborn
Machine Learning| Scikit-learn, XGBoost
Imbalanced Data Handling| SMOTE
Model Deployment| Streamlit, Render
Development Environment| Google Colab, VS Code

Machine Learning Workflow

1. Data loading and inspection
2. Exploratory Data Analysis
3. Missing-value handling and data preprocessing
4. Feature engineering and feature selection
5. Train-test split
6. Logistic Regression baseline model
7. Model development and comparison
8. Class imbalance handling using SMOTE
9. Model evaluation
10. Streamlit application development
11. Deployment on Render

Model Performance

The Logistic Regression model was used as the baseline. XGBoost with SMOTE was selected as the final model based on its evaluation performance.

Evaluation Metric| XGBoost Model
Test Accuracy| 61.64%
Macro F1-score Improvement| 19.4% relative improvement
Weighted F1-score| 0.5895

The model was evaluated using accuracy, Macro F1-score, Weighted F1-score, and class-wise performance metrics.

Macro F1-score gives equal importance to each CSAT class, making it useful for evaluating performance on an imbalanced dataset.

Key Insights

- The CSAT target variable is highly imbalanced, with score 5 representing the majority class.
- Accuracy alone may not adequately represent model performance across all CSAT classes.
- Macro F1-score provides a more balanced view of classification performance across minority and majority classes.
- SMOTE was applied to address class imbalance during model training.
- Class-wise evaluation helps assess the model's ability to identify dissatisfied customers.

Project Structure

Flipkart-CSAT-Score-Prediction/
│
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── models/                 # Saved model and preprocessing files
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── notebooks/
│   └── EDA.ipynb           # Exploratory Data Analysis notebook
│
└── .gitignore

«Update the file names and folder structure according to your actual repository.»

Installation and Local Setup

1. Clone the repository

git clone https://github.com/YOUR_USERNAME/Flipkart-CSAT-Score-Prediction.git

2. Navigate to the project directory

cd Flipkart-CSAT-Score-Prediction

3. Create a virtual environment

python -m venv venv

Activate the environment:

Windows

venv\Scripts\activate

Mac/Linux

source venv/bin/activate

4. Install dependencies

pip install -r requirements.txt

5. Run the Streamlit application

streamlit run app.py

The application will open in your browser at:

http://localhost:8501

Deployment on Render

The Streamlit application can be deployed using Render.

Deployment Steps

1. Push the project files to your GitHub repository.
2. Sign in to "Render" (https://render.com/).
3. Select New + → Web Service.
4. Connect your GitHub repository.
5. Configure the service:

Setting| Value
Runtime| Python
Build Command| "pip install -r requirements.txt"
Start Command| "streamlit run app.py --server.port $PORT --server.address 0.0.0.0"

6. Click Deploy Web Service.
7. Once the deployment is successful, open the generated Render URL to access the application.

Future Improvements

- Improve minority-class prediction performance.
- Perform hyperparameter tuning and model optimization.
- Add explainability using SHAP.
- Enhance the Streamlit dashboard with additional visualizations.
- Monitor model performance and prediction patterns.

Author

Tanuj Verma

Aspiring Data Scientist | Machine Learning | Generative AI

GitHub: https://github.com/tanujver

Acknowledgements

This project was developed as part of hands-on Machine Learning practice, focusing on customer satisfaction prediction, imbalanced classification, model evaluation, and deployment.
