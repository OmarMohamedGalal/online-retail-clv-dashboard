# Online Retail Analysis & CLV Dashboard

An interactive Streamlit dashboard for exploring the UCI Online Retail transaction data and modelling customer lifetime value (CLV).

## What it does

- Cleans cancelled, invalid, duplicate, and incomplete transactions
- Displays monthly revenue, country performance, high-value customers, and top products
- Builds customer-level RFM-inspired features
- Compares Linear Regression and Random Forest CLV models

## Run locally

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the dashboard:

   ```bash
   streamlit run st_route.py
   ```

4. Upload the `Online Retail.xlsx` workbook in the sidebar. The dataset is deliberately excluded from Git because it is a large source-data file. You can obtain the original data from the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail).

## Project files

- `st_route.py` — portable Streamlit application
- `route_project[1].ipynb` — original exploratory notebook

## Notes

CLV here is historical customer spend. The dashboard’s modelling section is intended as an educational regression exercise; forecasting future customer value requires a time-based target and validation split.
