# Real Estate Investment Advisor – Predicting Property Profitability & Future Value

## Project Overview
This project is a Machine Learning-based Real Estate Investment Advisor designed to analyze property data, evaluate investment potential, and estimate future property values.

The project uses Python, Random Forest, MLflow, and Streamlit to build an interactive property investment analysis application.

## Objectives
- Analyze Indian real estate property data.
- Evaluate property investment potential.
- Train a Random Forest classification model.
- Track model parameters, metrics, and artifacts using MLflow.
- Estimate future property value based on an assumed annual growth rate.
- Present insights through an interactive Streamlit application.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Random Forest
- MLflow
- Streamlit
- Plotly

## Project Structure
- app.py
- real estate project.py
- real estate project.ipynb
- investment_model.pkl
- requirements.txt

## Machine Learning
A Random Forest Classifier is used to classify properties based on the project's investment-label criteria.

Model parameters and evaluation metrics are tracked using MLflow.

## Future Property Value Estimation
The application estimates future property value using an assumed annual growth rate of 8% over a five-year period. This is a scenario-based estimate, not a guaranteed market forecast.

## How to Run the Project

1. Clone this repository:

   ```bash
   git clone https://github.com/athirasudhakaranofficials24-wq/Real-Estate-Investment-Advisor.git
   ```

2. Navigate to the project folder:

   ```bash
   cd Real-Estate-Investment-Advisor
   ```

3. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

## Model Tracking with MLflow
MLflow is used to track:
- Model parameters
- Model accuracy
- Trained model artifacts

## Disclaimer
Investment classifications depend on the dataset and the rules used to create the target variable. Future property values are illustrative estimates based on the assumed growth rate and should not be treated as financial advice.

## Author
**Athira A S**

GitHub: [athirasudhakaranofficials24-wq](https://github.com/athirasudhakaranofficials24-wq)
