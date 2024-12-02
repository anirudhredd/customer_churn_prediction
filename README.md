Customer Churn Prediction
This project focuses on predicting customer churn in the banking sector using machine learning models like Random Forest and Gradient Boosting. It identifies key churn drivers and provides actionable insights to enhance customer retention strategies.

Project Overview
Objective: Predict and analyze customer churn, and provide data-driven recommendations for improving customer retention.
Dataset: Churn_Modelling.csv, consisting of 10,000 rows with features like geography, age, balance, and churn status.
Tools Used: R, Random Forest, Gradient Boosting, Logistic Regression, ggplot2, SHAP.
Repository Structure
bash
Copy code
customer_churn_prediction/
├── data/
│   └── Churn_Modelling.csv             # Cleaned dataset for analysis
├── scripts/
│   └── Project_Models.R                # R script for predictive modeling
├── docs/
│   ├── Project_Report.pdf              # Detailed report with methodology and insights
│   ├── Presentation.pptx               # Project presentation
│   └── Proposal.docx                   # Initial project proposal
├── README.md                           # Overview of the project
Key Features
Data Analysis:

Preprocessed categorical features like Geography and Gender.
Explored correlations using heatmaps and visualizations.
Predictive Modeling:

Logistic Regression for baseline comparison.
Random Forest and Gradient Boosting for advanced modeling.
Evaluated models using ROC-AUC, accuracy, and feature importance.
Insights:

Key churn predictors include age, geography, and balance.
Ensemble models (e.g., Gradient Boosting) achieved the highest AUC.
Business Recommendations:

Target high-risk segments with low balances.
Implement regional strategies based on churn rates by geography.
How to Use
Clone the repository:
bash
Copy code
git clone https://github.com/your-username/customer_churn_prediction.git
Navigate to the scripts/ folder and run Project_Models.R to replicate the analysis.
View the cleaned dataset in the data/ folder for exploratory analysis.
Review the detailed report and presentation in the docs/ folder for insights.
Results
Gradient Boosting achieved an AUC of 0.7139, outperforming other models.
Key predictors include:
Age: Younger customers are more likely to churn.
Geography: Regional differences affect churn rates.
Balance: Low account balances correlate with higher churn.
