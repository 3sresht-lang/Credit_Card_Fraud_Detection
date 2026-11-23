
# ============================================================
# CREDIT CARD FRAUD DETECTION - COMPLETE STREAMLIT DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #101827;
}

[data-testid="stMetric"] {
    background-color: #1E293B;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #334155;
}

h1, h2, h3 {
    color: #38BDF8;
}

.stButton button {
    background-color: #2563EB;
    color: white;
    border-radius: 8px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# CUSTOMER SAMPLE INFORMATION
# ============================================================

first_names = [
    "Aarav", "Indhu", "Rahul", "Priya", "Ananya",
    "Arjun", "Sneha", "Kiran", "Divya", "Rohan",
    "Meera", "Vikram", "Kavya", "Aditya", "Pooja",
    "Nisha", "Varun", "Neha", "Sanjay", "Aisha"
]

last_names = [
    "Sharma", "Kumar", "Patel", "Reddy", "Rao",
    "Singh", "Das", "Mehta", "Verma", "Nair"
]

def add_customer_information(data):

    data = data.copy()

    rng = np.random.default_rng(42)

    data["Customer_Name"] = [
        first_names[i] + " " + last_names[j]
        for i, j in zip(
            rng.integers(0, len(first_names), len(data)),
            rng.integers(0, len(last_names), len(data))
        )
    ]

    data["Customer_ID"] = [
        f"CUST{100001 + i}"
        for i in range(len(data))
    ]

    return data

# ============================================================
# DATA LOADING
# ============================================================

def prepare_data(data):

    data.columns = data.columns.str.strip()

    if "Class" not in data.columns:
        raise ValueError(
            "Dataset must contain a Class column."
        )

    if "Amount" not in data.columns:
        raise ValueError(
            "Dataset must contain an Amount column."
        )

    data["Class"] = pd.to_numeric(
        data["Class"], errors="coerce"
    )

    data["Amount"] = pd.to_numeric(
        data["Amount"], errors="coerce"
    )

    data = data.dropna(subset=["Class", "Amount"])

    data = data[data["Class"].isin([0, 1])]

    data["Class"] = data["Class"].astype(int)

    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns

    data[numeric_columns] = data[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    data = data.dropna(subset=numeric_columns)

    if data.empty:
        raise ValueError("No valid transaction data found.")

    return data.reset_index(drop=True)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("💳 Fraud Detection")

st.sidebar.markdown("---")

st.sidebar.subheader("Upload Transaction Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

st.sidebar.markdown("---")

st.sidebar.subheader("Navigation")

page = st.sidebar.radio(
    "Choose Section",
    [
        "Overview",
        "Transactions",
        "Customer Details",
        "Fraud Analysis",
        "Predict Fraud",
        "Model Performance",
        "Chat Assistant",
        "About Project"
    ]
)

# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data(show_spinner="Loading transaction data...")
def load_csv(file_bytes):

    import io

    data = pd.read_csv(io.BytesIO(file_bytes))

    return prepare_data(data)

@st.cache_data(show_spinner="Loading local dataset...")
def load_local_csv():

    data = pd.read_csv("creditcard.csv")

    return prepare_data(data)

try:

    if uploaded_file is not None:

        raw_df = load_csv(uploaded_file.getvalue())

    else:

        raw_df = load_local_csv()

    df = add_customer_information(raw_df)

    st.sidebar.success("Dataset loaded successfully!")

except FileNotFoundError:

    st.error(
        "creditcard.csv was not found. "
        "Place it in the same folder as app.py "
        "or upload it using the sidebar."
    )

    st.stop()

except Exception as error:

    st.error(f"Dataset loading error: {error}")

    st.stop()

# ============================================================
# BASIC STATISTICS
# ============================================================

total_transactions = len(df)

fraud_transactions = int(
    (df["Class"] == 1).sum()
)

genuine_transactions = int(
    (df["Class"] == 0).sum()
)

fraud_percentage = (
    fraud_transactions / total_transactions * 100
)

total_amount = df["Amount"].sum()

average_amount = df["Amount"].mean()

# ============================================================
# TIME FEATURES
# ============================================================

if "Time" in df.columns:

    df["Elapsed_Hour"] = (
        df["Time"] // 3600
    ).astype(int)

else:

    df["Elapsed_Hour"] = 0

# ============================================================
# HEADER
# ============================================================

st.title("💳 CREDIT CARD FRAUD DETECTION")

st.markdown(
    "### Machine Learning Based Transaction Monitoring System"
)

st.caption(
    "Real-time style transaction analytics | "
    "Fraud monitoring | Random Forest classification"
)

st.markdown("---")

# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.header("📊 Dashboard Overview")

    # KPI CARDS

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    c2.metric(
        "Fraud Transactions",
        f"{fraud_transactions:,}"
    )

    c3.metric(
        "Genuine Transactions",
        f"{genuine_transactions:,}"
    )

    c4.metric(
        "Fraud Percentage",
        f"{fraud_percentage:.3f}%"
    )

    st.markdown("---")

    c5, c6 = st.columns(2)

    c5.metric(
        "Total Transaction Amount",
        f"${total_amount:,.2f}"
    )

    c6.metric(
        "Average Transaction Amount",
        f"${average_amount:,.2f}"
    )

    st.markdown("---")

    st.subheader("📈 Transaction Analytics - 10 Charts")

    # CHART 1: FRAUD DISTRIBUTION

    st.markdown("#### 1. Fraud vs Genuine Transactions")

    fig1 = px.pie(
        df,
        names="Class",
        hole=0.55,
        title="Transaction Distribution",
        color="Class",
        color_discrete_map={
            0: "#22C55E",
            1: "#EF4444"
        }
    )

    fig1.update_traces(
        textinfo="label+percent"
    )

    st.plotly_chart(fig1, use_container_width=True)

    # CHART 2: TRANSACTION COUNT

    st.markdown("#### 2. Transaction Count")

    count_df = df["Class"].value_counts().reset_index()

    count_df.columns = ["Transaction Type", "Count"]

    count_df["Transaction Type"] = count_df[
        "Transaction Type"
    ].map({
        0: "Genuine",
        1: "Fraud"
    })

    fig2 = px.bar(
        count_df,
        x="Transaction Type",
        y="Count",
        color="Transaction Type",
        title="Number of Transactions",
        color_discrete_map={
            "Genuine": "#22C55E",
            "Fraud": "#EF4444"
        },
        text_auto=True
    )

    st.plotly_chart(fig2, use_container_width=True)

    # CHART 3: AMOUNT DISTRIBUTION

    st.markdown("#### 3. Transaction Amount Distribution")

    fig3 = px.histogram(
        df,
        x="Amount",
        nbins=50,
        title="Distribution of Transaction Amounts",
        color_discrete_sequence=["#38BDF8"]
    )

    st.plotly_chart(fig3, use_container_width=True)

    # CHART 4: AMOUNT BY FRAUD CLASS

    st.markdown("#### 4. Transaction Amount by Class")

    fig4 = px.box(
        df,
        x="Class",
        y="Amount",
        color="Class",
        title="Amount Comparison",
        color_discrete_map={
            0: "#22C55E",
            1: "#EF4444"
        }
    )

    st.plotly_chart(fig4, use_container_width=True)

    # CHART 5: TRANSACTIONS BY HOUR

    st.markdown("#### 5. Transactions by Elapsed Hour")

    hour_data = (
        df.groupby("Elapsed_Hour")
        .size()
        .reset_index(name="Transactions")
    )

    fig5 = px.line(
        hour_data,
        x="Elapsed_Hour",
        y="Transactions",
        markers=True,
        title="Transactions Over Elapsed Hours"
    )

    st.plotly_chart(fig5, use_container_width=True)

    # CHART 6: FRAUD BY ELAPSED HOUR

    st.markdown("#### 6. Fraud Transactions by Elapsed Hour")

    fraud_hour = (
        df[df["Class"] == 1]
        .groupby("Elapsed_Hour")
        .size()
        .reset_index(name="Fraud Count")
    )

    fig6 = px.bar(
        fraud_hour,
        x="Elapsed_Hour",
        y="Fraud Count",
        title="Fraud Activity by Elapsed Hour",
        color_discrete_sequence=["#EF4444"]
    )

    st.plotly_chart(fig6, use_container_width=True)

    # CHART 7: TIME DISTRIBUTION

    st.markdown("#### 7. Transaction Time Distribution")

    if "Time" in df.columns:

        fig7 = px.histogram(
            df,
            x="Time",
            nbins=40,
            title="Transaction Time Distribution",
            color_discrete_sequence=["#A855F7"]
        )

        st.plotly_chart(
            fig7,
            use_container_width=True
        )

    else:

        st.info("Time column is unavailable.")

    # CHART 8: AMOUNT VS TIME

    st.markdown("#### 8. Amount vs Time")

    plot_data = df.sample(
        min(5000, len(df)),
        random_state=42
    )

    if "Time" in df.columns:

        fig8 = px.scatter(
            plot_data,
            x="Time",
            y="Amount",
            color="Class",
            title="Transaction Amount vs Time",
            color_discrete_map={
                0: "#22C55E",
                1: "#EF4444"
            }
        )

        st.plotly_chart(
            fig8,
            use_container_width=True
        )

    # CHART 9: CORRELATION HEATMAP

    st.markdown("#### 9. Feature Correlation Heatmap")

    correlation_columns = [
        c for c in [
            "Time", "Amount", "Class",
            "V1", "V2", "V3", "V4", "V5"
        ]
        if c in df.columns
    ]

    correlation = df[
        correlation_columns
    ].corr()

    fig9 = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        title="Correlation Between Dataset Features",
        color_continuous_scale="RdBu_r"
    )

    st.plotly_chart(
        fig9,
        use_container_width=True
    )

    # CHART 10: FEATURE CORRELATION WITH FRAUD

    st.markdown("#### 10. Features Related to Fraud Class")

    feature_columns = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in ["Class", "Elapsed_Hour"]
    ]

    feature_corr = (
        df[feature_columns + ["Class"]]
        .corr()["Class"]
        .drop("Class")
        .abs()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    feature_corr.columns = [
        "Feature",
        "Absolute Correlation"
    ]

    fig10 = px.bar(
        feature_corr,
        x="Absolute Correlation",
        y="Feature",
        orientation="h",
        title="Top 10 Features Associated with Fraud",
        color="Absolute Correlation",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(
        fig10,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Recent Transactions")

    st.dataframe(
        df[[
            "Customer_ID",
            "Customer_Name",
            "Amount",
            "Class"
        ]].head(15),
        use_container_width=True
    )

# ============================================================
# TRANSACTIONS PAGE
# ============================================================

elif page == "Transactions":

    st.header("💳 Transaction Management")

    transaction_type = st.selectbox(
        "Filter Transactions",
        [
            "All",
            "Genuine",
            "Fraud"
        ]
    )

    if transaction_type == "Fraud":

        filtered_df = df[df["Class"] == 1]

    elif transaction_type == "Genuine":

        filtered_df = df[df["Class"] == 0]

    else:

        filtered_df = df

    st.write(
        f"Showing {len(filtered_df):,} transactions"
    )

    search = st.text_input(
        "Search Customer Name or ID"
    )

    if search:

        filtered_df = filtered_df[
            filtered_df["Customer_Name"].str.contains(
                search, case=False, na=False
            )
            |
            filtered_df["Customer_ID"].str.contains(
                search, case=False, na=False
            )
        ]

    st.dataframe(
        filtered_df.head(1000),
        use_container_width=True
    )

    csv_data = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Filtered Transactions",
        csv_data,
        "filtered_transactions.csv",
        "text/csv"
    )

# ============================================================
# CUSTOMER DETAILS
# ============================================================

elif page == "Customer Details":

    st.header("👥 Customer Information")

    st.info(
        "Customer names and IDs below are synthetic "
        "demonstration values, not original Kaggle identities."
    )

    customer_data = df[[
        "Customer_ID",
        "Customer_Name",
        "Amount",
        "Class"
    ]].copy()

    customer_data["Status"] = customer_data[
        "Class"
    ].map({
        0: "Genuine",
        1: "Fraud"
    })

    st.dataframe(
        customer_data.head(1000),
        use_container_width=True
    )

    st.subheader("Customer Transaction Summary")

    selected_customer = st.selectbox(
        "Select Customer",
        customer_data["Customer_ID"].head(1000)
    )

    customer = customer_data[
        customer_data["Customer_ID"] == selected_customer
    ]

    st.dataframe(customer)

    st.metric(
        "Customer Transaction Amount",
        f"${customer['Amount'].sum():,.2f}"
    )

# ============================================================
# FRAUD ANALYSIS
# ============================================================

elif page == "Fraud Analysis":

    st.header("🚨 Fraud Analysis")

    fraud_df = df[df["Class"] == 1]

    genuine_df = df[df["Class"] == 0]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Fraud Count",
        f"{len(fraud_df):,}"
    )

    c2.metric(
        "Fraud Amount",
        f"${fraud_df['Amount'].sum():,.2f}"
    )

    c3.metric(
        "Average Fraud Amount",
        f"${fraud_df['Amount'].mean():,.2f}"
    )

    st.subheader("Fraud Transaction Records")

    st.dataframe(
        fraud_df[[
            "Customer_ID",
            "Customer_Name",
            "Amount",
            "Class"
        ]].head(500),
        use_container_width=True
    )

    st.subheader("Fraud Amount Analysis")

    fig = px.histogram(
        fraud_df,
        x="Amount",
        nbins=30,
        title="Fraud Transaction Amounts",
        color_discrete_sequence=["#EF4444"]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Fraud vs Genuine Average Amount")

    comparison = pd.DataFrame({
        "Type": ["Genuine", "Fraud"],
        "Average Amount": [
            genuine_df["Amount"].mean(),
            fraud_df["Amount"].mean()
        ]
    })

    fig = px.bar(
        comparison,
        x="Type",
        y="Average Amount",
        color="Type",
        title="Average Transaction Amount Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

def train_fraud_model(data):

    feature_columns = [
        c for c in data.select_dtypes(include=np.number).columns
        if c not in ["Class", "Elapsed_Hour"]
    ]

    if not feature_columns:
        raise ValueError("No numerical features found.")

    model_data = data[
        feature_columns + ["Class"]
    ].copy()

    # Limit training size to make it practical
    # while retaining all minority-class examples.

    fraud_data = model_data[
        model_data["Class"] == 1
    ]

    genuine_data = model_data[
        model_data["Class"] == 0
    ]

    max_genuine = 50000

    if len(genuine_data) > max_genuine:

        genuine_data = genuine_data.sample(
            n=max_genuine,
            random_state=42
        )

    balanced_data = pd.concat(
        [genuine_data, fraud_data],
        ignore_index=True
    )

    X = balanced_data[feature_columns]

    y = balanced_data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(
            y_test, predictions
        ),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        )
    }

    return (
        model,
        feature_columns,
        X_test,
        y_test,
        predictions,
        metrics
    )

# ============================================================
# PREDICT FRAUD PAGE
# ============================================================

if page == "Predict Fraud":

    st.header("🤖 Machine Learning Fraud Prediction")

    st.write(
        "Train a Random Forest model using the uploaded transaction data."
    )

    if st.button("🚀 Train Machine Learning Model"):

        with st.spinner(
            "Training Random Forest model. Please wait..."
        ):

            try:

                result = train_fraud_model(df)

                st.session_state["model_result"] = result

                st.success(
                    "Machine Learning Model Trained Successfully!"
                )

            except Exception as error:

                st.error(f"Model training failed: {error}")

    if "model_result" in st.session_state:

        (
            model,
            features,
            X_test,
            y_test,
            predictions,
            metrics
        ) = st.session_state["model_result"]

        st.subheader("Model Evaluation")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Accuracy",
            f"{metrics['Accuracy'] * 100:.2f}%"
        )

        m2.metric(
            "Precision",
            f"{metrics['Precision'] * 100:.2f}%"
        )

        m3.metric(
            "Recall",
            f"{metrics['Recall'] * 100:.2f}%"
        )

        m4.metric(
            "F1 Score",
            f"{metrics['F1 Score'] * 100:.2f}%"
        )

        st.markdown("---")

        st.subheader("Predict an Existing Transaction")

        row_number = st.number_input(
            "Enter Transaction Row Number",
            min_value=0,
            max_value=len(df) - 1,
            value=0,
            step=1
        )

        if st.button("🔍 Predict Selected Transaction"):

            selected = df.iloc[
                [int(row_number)]
            ][features]

            prediction = model.predict(selected)[0]

            probability = model.predict_proba(
                selected
            )[0][1]

            if prediction == 1:

                st.error(
                    "🚨 MODEL PREDICTION: POTENTIAL FRAUD"
                )

            else:

                st.success(
                    "✅ MODEL PREDICTION: GENUINE"
                )

            st.metric(
                "Estimated Fraud Probability",
                f"{probability * 100:.2f}%"
            )

            st.write(
                "Transaction Amount:",
                f"${df.iloc[int(row_number)]['Amount']:.2f}"
            )

            st.caption(
                "This is a model prediction, not a confirmed fraud finding."
            )

        st.markdown("---")

        st.subheader("Feature Importance")

        importance = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values(
            "Importance",
            ascending=False
        ).head(15)

        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Most Important Model Features"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

elif page == "Model Performance":

    st.header("📈 Model Performance")

    if "model_result" not in st.session_state:

        st.warning(
            "Please train the model first in Predict Fraud."
        )

    else:

        (
            model,
            features,
            X_test,
            y_test,
            predictions,
            metrics
        ) = st.session_state["model_result"]

        st.subheader("Performance Metrics")

        metrics_df = pd.DataFrame(
            list(metrics.items()),
            columns=["Metric", "Score"]
        )

        st.dataframe(
            metrics_df,
            use_container_width=True
        )

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            predictions
        )

        fig = px.imshow(
            cm,
            text_auto=True,
            x=["Genuine", "Fraud"],
            y=["Genuine", "Fraud"],
            title="Confusion Matrix",
            color_continuous_scale="Blues"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0
        )

        st.dataframe(
            pd.DataFrame(report).transpose(),
            use_container_width=True
        )

# ============================================================
# CHAT ASSISTANT
# ============================================================

elif page == "Chat Assistant":

    st.header("💬 Fraud Detection Chat Assistant")

    st.write(
        "Ask questions about transactions, fraud, charts and the model."
    )

    if "chat_history" not in st.session_state:

        st.session_state.chat_history = []

    # DISPLAY OLD MESSAGES

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # USER QUESTION

    user_question = st.chat_input(
        "Ask me about fraud detection..."
    )

    if user_question:

        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        question = user_question.lower()

        if "total" in question or "transaction" in question:

            answer = (
                f"Your dataset contains {total_transactions:,} "
                f"transactions."
            )

        elif "fraud" in question:

            answer = (
                f"There are {fraud_transactions:,} fraud-labelled "
                f"transactions in your dataset."
            )

        elif "genuine" in question:

            answer = (
                f"There are {genuine_transactions:,} "
                f"genuine-labelled transactions."
            )

        elif "percentage" in question:

            answer = (
                f"The fraud percentage is {fraud_percentage:.3f}%."
            )

        elif "amount" in question:

            answer = (
                f"The total transaction amount is "
                f"${total_amount:,.2f}. "
                f"The average is ${average_amount:,.2f}."
            )

        elif "customer" in question or "name" in question:

            answer = (
                "The dashboard generates synthetic demonstration "
                "customer names and IDs because the original Kaggle "
                "dataset does not contain customer identities."
            )

        elif "model" in question or "prediction" in question:

            if "model_result" in st.session_state:

                result = st.session_state["model_result"]

                answer = (
                    "The Random Forest model has been trained. "
                    f"Its test accuracy is "
                    f"{result[5]['Accuracy'] * 100:.2f}%."
                )

            else:

                answer = (
                    "Please open Predict Fraud and train the "
                    "Random Forest model first."
                )

        elif "chart" in question:

            answer = (
                "The Overview page contains 10 charts including "
                "fraud distribution, transaction amounts, time "
                "analysis, correlation and feature analysis."
            )

        elif "hello" in question or "hi" in question:

            answer = (
                "Hello! Welcome to the Credit Card Fraud Detection "
                "Assistant. How can I help you?"
            )

        else:

            answer = (
                "I can answer questions about transaction counts, "
                "fraud, genuine transactions, amounts, customer "
                "demonstration information and model performance."
            )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

        st.rerun()

    if st.button("Clear Chat"):

        st.session_state.chat_history = []

        st.rerun()

# ============================================================
# ABOUT PROJECT PAGE
# ============================================================

elif page == "About Project":

    st.header("📚 About the Project")

    st.subheader("Credit Card Fraud Detection System")

    st.write("""
    This project uses machine learning and data analytics
    to identify potentially fraudulent credit card transactions.

    The system analyses transaction data, studies patterns,
    displays charts and predicts whether a transaction
    may be fraudulent.
    """)

    st.subheader("Technologies Used")

    technologies = [
        "Python",
        "Streamlit",
        "Pandas",
        "NumPy",
        "Plotly",
        "Scikit-learn",
        "Random Forest Classifier",
        "Machine Learning"
    ]

    for technology in technologies:

        st.write("✅", technology)

    st.subheader("Project Features")

    features_list = [
        "CSV Dataset Upload",
        "Transaction Analysis",
        "10 Interactive Charts",
        "Customer Demonstration Details",
        "Fraud Detection",
        "Machine Learning Predictions",
        "Model Performance Metrics",
        "Confusion Matrix",
        "Chat Assistant",
        "CSV Download"
    ]

    for feature in features_list:

        st.write("🔹", feature)

    st.subheader("Dataset Information")

    st.write(
        "Dataset: Kaggle Credit Card Fraud Detection"
    )

    st.write(
        "Target column: Class (0 = Genuine, 1 = Fraud)"
    )

    st.write(
        "Features: Time, V1 to V28, Amount and Class"
    )

    st.warning(
        "This is an educational demonstration system. "
        "Predictions should not be used as the sole basis "
        "for real financial decisions."
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <center>
    <h4>💳 Credit Card Fraud Detection Dashboard</h4>
    <p>Machine Learning Based Transaction Monitoring System</p>
    <p>Developed using Python, Streamlit and Scikit-learn</p>
    </center>
    """,
    unsafe_allow_html=True
)