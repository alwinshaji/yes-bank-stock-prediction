📈 Yes Bank Stock Price Prediction

Forecasting closing prices of Yes Bank stock using machine learning models.

🧰 Project Structure

yes-bank-stock-prediction/

├── data/
│ └── data_YesBank_StockPrices.csv # Historical stock dataset

├── model/
│ └── linear_regression_model.joblib # Trained Linear Regression model

├── notebooks/
│ └── yesbank_stock_prediction.ipynb # Notebook with code & outputs

├── reports/
│ └── yesbank_stock_prediction_final.pdf # Final report with charts & insights

└── README.md # Project overview


🎯 Objective

To build and evaluate machine learning models for predicting the closing price of Yes Bank stock, using historical data and a feature engineering pipeline.

🚀 Workflow Overview

1. Data Preparation 
   - Loaded and cleaned time-series data  
   - Engineered features like `Prev_Close`, rolling averages, etc.

2. Exploratory Data Analysis (EDA) 
   - Visualized price trends, volatility, and relationships  
   - Identified strong correlation between `Prev_Close` and `Close`

3. Model Development
   - Trained multiple models:  
     - Linear Regression — best performance (RMSE ≈ 3.08, R² ≈ 0.9995)  
     - Random Forest
     - Gradient Boosting (with & without GridSearch)

4. Hyperparameter Tuning
   - Used `GridSearchCV` on ensemble models to improve RMSE/R²

5. Model Selection & Deployment Prep
   - Finalized Linear Regression as the production model  
   - Saved model using `joblib` for future use  
   - Conducted a sanity check via loaded model prediction

 📊 Evaluation Metrics

- RMSE: Measures prediction error in stock price units (lower = better).  
- R² Score: Indicates the proportion of variance explained (closer to 1 = better).

Linear Regression significantly outperformed the others, making it the top choice.

🧩 How to Run & Use

1. Open the notebook in **Google Colab** for an interactive walkthrough:  
   👉 `https://colab.research.google.com/github/alwinshaji/yes-bank-stock-prediction/blob/main/notebooks/yesbank_stock_prediction.ipynb`

2. Download and inspect the final report PDF in `reports/` for full visual results.

3. Load the trained model with:
   ```python
   import joblib
   loaded_model = joblib.load('models/linear_regression_model.joblib')

Use it for prediction:

new_input = [[98.50]]  # previous close price
prediction = loaded_model.predict(new_input)
print(f"Predicted Close Price: ₹{prediction[0]:.2f}")


📚 Libraries Used

- pandas, numpy – Data management
- matplotlib, seaborn – Visualizations
- scikit-learn – Modeling & evaluation
- joblib – Model serialization

📝 Next Steps

- Add confidence intervals around the predictions
- Test on more recent data to evaluate robustness
- Integrate into a streamlit app or web API for real-time usage

⚠️ Notes
- The reports/ folder contains the full PDF output (including charts).
- For interactive exploration, use the notebook — charts and code are fully executable.


✅ How to Contribute

- Feel free to:
- Fork the repo
- Suggest improvements or features
- Submit issues or pull requests

Thanks for checking it out — and happy coding! 🚀
















   
