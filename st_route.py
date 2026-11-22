"""Interactive analysis and customer-value modelling for the Online Retail data set."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Online Retail CLV", layout="wide")

REQUIRED_COLUMNS = {"InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"}


@st.cache_data(show_spinner=False)
def read_data(source) -> pd.DataFrame:
    """Load and validate an Online Retail Excel workbook."""
    frame = pd.read_excel(source)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"The workbook is missing required columns: {', '.join(sorted(missing))}")
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"], errors="coerce")
    frame["TotalPrice"] = frame["Quantity"] * frame["UnitPrice"]
    return frame


def clean_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove cancellations/invalid orders and return a clean transaction table."""
    clean = frame.copy()
    clean = clean[~clean["InvoiceNo"].astype(str).str.startswith("C")]
    clean = clean[(clean["Quantity"] > 0) & (clean["UnitPrice"] > 0)]
    clean = clean.dropna(subset=["CustomerID", "InvoiceDate"])
    clean["Description"] = clean["Description"].fillna("Unknown")
    clean = clean.drop_duplicates()
    clean["TotalPrice"] = clean["Quantity"] * clean["UnitPrice"]
    return clean


def build_customer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build customer features and the historical-spend CLV target."""
    data = frame.sort_values(["CustomerID", "InvoiceDate"]).copy()
    reference_date = data["InvoiceDate"].max() + pd.Timedelta(days=1)
    return data.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda dates: (reference_date - dates.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        CLV=("TotalPrice", "sum"),
        AvgOrderValue=("TotalPrice", "mean"),
        AvgItemQuantity=("Quantity", "mean"),
        ActiveDays=("InvoiceDate", lambda dates: dates.dt.date.nunique()),
    ).reset_index()


def evaluate_models(customer_data: pd.DataFrame, random_state: int, test_size: float) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Fit baseline and random-forest regressors and return their test metrics."""
    features = customer_data.drop(columns=["CustomerID", "CLV"])
    target = customer_data["CLV"]
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=250, max_depth=12, min_samples_leaf=2, random_state=random_state, n_jobs=-1),
    }
    results, predictions = [], {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        predictions[name] = predicted
        results.append({"Model": name, "RMSE": mean_squared_error(y_test, predicted) ** 0.5, "MAE": mean_absolute_error(y_test, predicted), "R²": r2_score(y_test, predicted)})
    metrics = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    return metrics, y_test, predictions[metrics.loc[0, "Model"]]


def main() -> None:
    st.title("Online Retail Analysis & CLV Dashboard")
    st.caption("Explore retail transactions and estimate customer lifetime value.")
    default_file = Path(__file__).with_name("Online Retail.xlsx")
    upload = st.sidebar.file_uploader("Upload Online Retail workbook", type=["xlsx"])
    source = upload if upload is not None else (default_file if default_file.exists() else None)
    st.sidebar.caption("The bundled workbook is used when available; otherwise upload an .xlsx file.")
    if source is None:
        st.info("Upload the Online Retail Excel workbook to begin.")
        st.stop()
    try:
        raw = read_data(source)
    except (ValueError, OSError, TypeError) as error:
        st.error(f"Could not load the workbook: {error}")
        st.stop()
    cleaned = clean_transactions(raw)
    customer_data = build_customer_features(cleaned)
    st.sidebar.metric("Clean transactions", f"{len(cleaned):,}")
    st.sidebar.metric("Customers", f"{len(customer_data):,}")

    overview, customers, models = st.tabs(["Overview", "Customers & products", "CLV models"])
    with overview:
        col1, col2, col3 = st.columns(3)
        col1.metric("Revenue", f"£{cleaned['TotalPrice'].sum():,.0f}")
        col2.metric("Orders", f"{cleaned['InvoiceNo'].nunique():,}")
        col3.metric("Countries", f"{cleaned['Country'].nunique():,}")
        st.subheader("Monthly revenue")
        monthly = cleaned.set_index("InvoiceDate").resample("ME")["TotalPrice"].sum()
        st.line_chart(monthly)
        st.subheader("Top countries by revenue")
        st.dataframe(cleaned.groupby("Country", as_index=False)["TotalPrice"].sum().sort_values("TotalPrice", ascending=False).head(15), use_container_width=True)
    with customers:
        st.subheader("Top customers by CLV")
        st.dataframe(customer_data.nlargest(20, "CLV"), use_container_width=True)
        st.subheader("Top products by revenue")
        top_products = cleaned.groupby(["StockCode", "Description"], as_index=False)["TotalPrice"].sum().nlargest(15, "TotalPrice")
        fig, axis = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_products, x="TotalPrice", y="Description", ax=axis, color="#4C78A8")
        axis.set(xlabel="Revenue (£)", ylabel="")
        st.pyplot(fig, clear_figure=True)
    with models:
        st.write("The target is historical customer spend (CLV). Results are for exploratory learning, not future-value forecasting.")
        test_percent = st.slider("Test-set size (%)", 10, 40, 20, 5)
        if st.button("Train models", type="primary"):
            with st.spinner("Training models..."):
                metrics, actual, predicted = evaluate_models(customer_data, 42, test_percent / 100)
            st.dataframe(metrics.style.format({"RMSE": "£{:,.2f}", "MAE": "£{:,.2f}", "R²": "{:.3f}"}), use_container_width=True)
            fig, axis = plt.subplots(figsize=(6, 6))
            axis.scatter(actual, predicted, alpha=0.45)
            maximum = max(actual.max(), predicted.max())
            axis.plot([0, maximum], [0, maximum], "--", color="gray")
            axis.set(xlabel="Actual CLV (£)", ylabel="Predicted CLV (£)", title="Best model: actual vs predicted")
            st.pyplot(fig, clear_figure=True)


if __name__ == "__main__":
    main()
