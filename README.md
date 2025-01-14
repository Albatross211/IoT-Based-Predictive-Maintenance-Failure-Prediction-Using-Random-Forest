# IoT-Based-Predictive-Maintenance-Failure-Prediction-Using-Random-Forest
# IoT-Based Predictive Maintenance: Failure Prediction Using Random Forest

## 📋 Project Overview
This project aims to predict the remaining useful life (RUL) of industrial engines using sensor data and machine learning. The goal is to enable predictive maintenance by forecasting the time to failure, reducing downtime, and improving operational efficiency. A Random Forest model was developed to process IoT sensor data and classify failure conditions with high accuracy.

---

## 🔑 Key Features
- **Predictive Maintenance**: Predicts the time-to-failure (`time_to_failure`) of engines.
- **Data Preprocessing**: Includes data cleaning, normalization, and feature engineering.
- **Random Forest Algorithm**: Utilized for accurate and interpretable predictions.
- **Classification Threshold Optimization**: Fine-tuned the model for a balance between precision and recall.
- **Performance Metrics**: Achieved:
  - **F1 Score**: 0.86
  - **Precision**: 0.94
  - **Recall**: 0.79
  - **Accuracy**: 0.94

---

## 📊 Dataset
The project uses the **NASA Turbofan Engine Degradation Simulation Dataset**, a publicly available dataset designed for predictive maintenance research. It includes:
- Engine cycles (`cycle`)
- Sensor readings (`sensor1` to `sensor21`)
- Operating conditions (`operating_condition1`, `operating_condition2`, `operating_condition3`)

---

## 🚀 Workflow
1. **Data Preprocessing**:
   - Removed redundant columns.
   - Normalized sensor data using MinMaxScaler.
   - Added `time_to_failure` as the target variable.

2. **Model Training**:
   - Trained a Linear Regression model as a baseline.
   - Optimized and implemented a Random Forest model for better performance.

3. **Evaluation**:
   - Metrics: F1 Score, Precision, Recall, Accuracy.
   - Visualized predictions and feature importance.

4. **Optimization**:
   - Hyperparameter tuning using GridSearchCV.
   - Adjusted classification thresholds for improved recall.

---

## 📈 Results
- Random Forest outperformed the baseline model with:
  - **Mean Absolute Error (MAE)**: 29.56
  - **Mean Squared Error (MSE)**: 1712.88
  - **R² Score**: 0.625
- Feature importance analysis revealed critical sensors impacting predictions.

---

## 🔧 Tools and Technologies
- **Programming Language**: Python
- **Libraries**:
  - `pandas`, `numpy` for data manipulation.
  - `scikit-learn` for model building and evaluation.
  - `matplotlib` for visualization.
  - `imblearn` for oversampling (SMOTE).
- **Platform**: Google Colab

---

## 🏆 Learning Outcomes
- Improved understanding of predictive maintenance concepts.
- Hands-on experience with Random Forest and regression-to-classification tasks.
- Gained skills in data preprocessing, hyperparameter tuning, and model evaluation.

---

## 🔮 Future Work
- Experiment with advanced models like XGBoost and LightGBM for improved performance.
- Extend the project to multi-class classification or continuous RUL prediction.
- Deploy the model using Flask or FastAPI for real-time predictions.

---

## 🤝 Acknowledgments
- **Dataset**: [NASA Prognostics Center of Excellence](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
- **Inspiration**: Research in IoT and predictive maintenance.

---

## 📬 Contact
Feel free to connect with me for any questions or collaborations:
- **Email**: hritikkapoor.work@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/hritik-kapoor/
