📊 Student Performance Analysis & Prediction
📌 Project Overview

This project analyzes and predicts student academic performance using demographic, academic, family, and lifestyle-related factors. The main objective is to understand which factors influence final grades and to build a realistic early-stage prediction model rather than an over-optimistic one.

The project follows a complete data science workflow: data understanding, exploratory data analysis (EDA), preprocessing, visualization, and machine learning modeling.

📂 Dataset Description

The dataset contains information about students, including:

Demographics: age, gender, school, address

Family background: parental education, family size, family relationship quality, family support

Academic behavior: study time, failures, absences

Lifestyle factors: going out, alcohol consumption, health, internet access

Target variable: G3 (final grade, 0–20)

⚠️ Intermediate grades (G1, G2) are present in the dataset but were excluded from modeling to avoid data leakage and ensure realistic predictions.

🔍 Exploratory Data Analysis (EDA)

EDA was performed using Matplotlib to extract meaningful insights. Key observations include:

Moderate study time leads to better performance than extremely low or high study time

Higher absences strongly correlate with lower final grades

Weekday alcohol consumption negatively impacts academic performance

Better family relationships are associated with more consistent and higher grades

Gender distribution is balanced and shows no strong standalone impact

These insights guided feature selection and model choice.

🧠 Machine Learning Approach

Problem Type: Regression

Target Variable: Final Grade (G3)

Model Used: Random Forest Regressor

Random Forest Regression was chosen because it:

Handles non-linear relationships effectively

Works well with mixed feature types

Is robust to noisy human-behavior data

Provides interpretability through feature importance

📈 Model Evaluation

The model was evaluated using regression metrics:

R² Score

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

The results show moderate predictive performance, which is expected when predicting final grades without prior exam scores. This reflects a realistic early-intervention scenario rather than an artificially inflated model.

⚖️ Project Philosophy

This project prioritizes:

Honest modeling over inflated accuracy

Avoidance of data leakage

Interpretability and real-world applicability

Rather than maximizing scores using future information, the model focuses on early prediction and insight generation.

🛠️ Tech Stack

Python

Pandas

Matplotlib

Scikit-learn

🚀 Future Improvements

One-hot encoding of nominal categorical features

Hyperparameter tuning for improved performance

Feature importance visualization

Comparison with baseline models (Linear Regression)

Early-risk student identification system

✅ Conclusion

This project demonstrates a realistic and responsible application of machine learning to educational data. It highlights how behavior, attendance, and support systems influence academic outcomes and provides a strong foundation for further educational analytics work.
