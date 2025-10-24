# Supply Chain Profitability Analysis & Predictive Dashboard

## Project Overview

This repository contains the complete workflow for a data science project focused on supply chain profitability. It leverages machine learning to model complex operational data, identifies key financial drivers, and provides strategic, data-driven recommendations. The project culminates in an interactive web dashboard for non-technical stakeholder engagement and decision support.

This analysis moves beyond traditional descriptive analytics (what happened) to predictive (what will happen) and prescriptive (what we should do) insights, providing a robust framework for strategic planning.

## Business Objective

The primary objective is to dissect the complex relationship between operational metrics (e.g., cost, efficiency, lead time) and financial performance (profit margin).

The project aims to:

1.  **Identify and quantify** the most significant drivers of profitability.
2.  **Develop a predictive model** to accurately forecast profit margin based on these drivers.
3.  **Provide actionable, strategic recommendations** to optimize costs, rebalance portfolios, and enhance business resilience.
4.  **Deliver an interactive tool** (Streamlit Dashboard) to make these complex insights accessible and explorable for all decision-makers.

## Methodology

The project follows a structured data science methodology, from data ingestion to strategic deployment.

1.  **Data Ingestion & Preparation**

      * Imported and merged raw datasets covering cost, revenue, supplier, and operational metrics.
      * Conducted rigorous data cleaning, handling missing values, and standardizing variables for consistency.
      * Performed **feature engineering** to create high-impact variables that are not present in the raw data, such as:
          * `Cost Ratio` (Total Costs / Revenue)
          * `Cost Efficiency` (Revenue / Total Costs)
          * `Profit Margin` ((Revenue - Total Costs) / Revenue)

2.  **Exploratory Data Analysis (EDA)**

      * Conducted a deep-dive EDA to uncover initial trends, cross-correlations (via heatmaps), and variable distributions.
      * Visualized product- and supplier-level performance to identify high- and low-performing segments.
      * This phase was crucial for hypothesis generation before modeling.

3.  **Predictive Modeling**

      * Utilized a `RandomForestRegressor` model, chosen for its high performance, robustness to non-linear relationships, and built-in feature importance interpretation.
      * The model was trained to predict `Profit Margin` based on all engineered and raw operational features.
      * Model performance was assessed using standard regression metrics (e.g., R-squared, Mean Absolute Error) to ensure predictive accuracy and reliability.

4.  **Simulation & Analysis**

      * **Sensitivity Analysis:** Executed a sensitivity analysis by simulating ±10% changes in key drivers (e.g., Cost Ratio, Price) to measure their elasticity and isolated impact on profitability.
      * **Scenario Modeling:** Developed three distinct business scenarios (Optimistic, Moderate, Pessimistic) to stress-test the business model against potential market shifts and operational changes.

5.  **Insights & Deployment**

      * **Strategic Formulation:** Translated model outputs (especially feature importances) and simulation results into a formal business insights report.
      * **Dashboard Development:** Engineered an interactive dashboard using Streamlit to serve as the project's front-end. This tool allows stakeholders to dynamically explore data, test scenarios, and internalize the key findings without technical expertise.

## Key Outcomes & Insights

The analysis and modeling process yielded several critical, high-value insights:

  * **Dominant Profit Drivers:** The analysis definitively identified `Cost Ratio` (54.5% importance) and `Cost Efficiency` (43.7% importance) as the two overwhelming drivers of profitability. Traditional metrics like order volume had a negligible impact, shifting the strategic focus from "selling more" to "selling more efficiently."
  * **Portfolio Optimization Opportunity:** Discovered that the 'Cosmetics' category, while not the highest revenue generator, delivers the best profit margins. This highlights a significant opportunity for strategic product mix optimization to improve overall portfolio profitability.
  * **Business Model Resilience:** Both sensitivity and scenario analyses confirmed a highly stable, high-margin business model. Profitability showed minimal volatility even under pessimistic scenarios, indicating strong operational insulation. The primary threat is cost inflation, not demand fluctuation.

## Strategic Recommendations

Based on the analysis, the following data-driven strategies are recommended:

1.  **Prioritize Cost Containment:** Focus on procurement efficiency, supplier negotiation, and logistics optimization, as these actions have the most direct and significant impact on margin.
2.  **Rebalance Product Portfolio:** Shift marketing and inventory focus toward higher-margin categories (Cosmetics) to improve the profit-per-sale average.
3.  **Strengthen Supplier Partnerships:** Consolidate volume with high-performing, low-defect suppliers (identified in the analysis) to enhance predictability and reduce cost variance.
4.  **Focus on Resilience, Not Just Optimization:** Given the existing high margins, the strategic focus should shift from incremental optimization to defensive resilience. This includes supply chain diversification and cost hedging to protect against market shocks.

## Technology Stack

  * **Data Analysis & Manipulation:** `pandas`, `numpy`
  * **Data Visualization:** `matplotlib`, `seaborn`, `plotly` (for interactive plots)
  * **Machine Learning:** `scikit-learn` (for `RandomForestRegressor`, `train_test_split`, `StandardScaler`)
  * **Dashboard & Web Framework:** `Streamlit` (for rapid, data-centric web app development)
  * **Environment:** `Python 3.11+`, `Jupyter Notebook` (for analysis & modeling), `VS Code` (for app development)

## Repository Structure

```
.
├── app.py                     # Main Streamlit dashboard application
├── Supply Chain.ipynb         # Jupyter Notebook with full analysis, modeling, and EDA
├── data/
│   ├── raw_data.csv           # Original raw dataset
│   └── processed_data.csv     # Cleaned, processed, and feature-engineered data
├── insights/
│   └── business_insights.pdf  # Final summary report with insights and recommendations
├── requirements.txt           # Python dependencies for reproducibility
└── README.md                  # Project documentation (this file)
```

## How to Run Locally

To explore the data or run the interactive dashboard, follow these steps.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/supply-chain-profitability.git
    cd supply-chain-profitability
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate   # On Mac/Linux
    .venv\Scripts\activate      # On Windows
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch the Streamlit dashboard:**

    ```bash
    streamlit run app.py
    ```

    The application will open in your default web browser.

## Conclusion

This project serves as an end-to-end example of applied data science for business strategy. It demonstrates the complete workflow from raw data ingestion and feature engineering to predictive modeling, simulation, and deployment in a stakeholder-ready format. By moving beyond descriptive analytics to predictive and prescriptive insights, this framework provides a robust, data-driven foundation for enhancing supply chain profitability and resilience.
