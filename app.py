import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_diabetes
import io

# Streamlit App Title
st.title("ML Playground: Upload & Train Your Model")

# Sidebar: Choose dataset source
dataset_source = st.sidebar.radio("Choose Data Source", ["Upload CSV", "Select UCI Dataset", "Custom URL"])

# Function to load UCI datasets
def load_data(dataset_name):
    if dataset_name == "Iris":
        data = load_iris()
    elif dataset_name == "Breast Cancer":
        data = load_breast_cancer()
    elif dataset_name == "Wine":
        data = load_wine()
    elif dataset_name == "Diabetes":
        data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df, data.target_names if hasattr(data, 'target_names') else None

df = None  # Initialize dataframe
target_names = None

# Handling different dataset sources
if dataset_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
elif dataset_source == "Select UCI Dataset":
    dataset_name = st.sidebar.selectbox("Select UCI Dataset", ["Iris", "Breast Cancer", "Wine", "Diabetes"])
    df, target_names = load_data(dataset_name)
elif dataset_source == "Custom URL":
    dataset_url = st.sidebar.text_input("Enter CSV URL")
    if dataset_url:
        df = pd.read_csv(dataset_url)

# Display dataset if loaded
if df is not None:
    st.write("### Dataset Preview")
    st.dataframe(df.head())
    
    st.write("### Dataset Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())
    
    st.write("### Feature Value Counts")
    st.write(df.nunique())
    
    # Feature Selection
    selected_features = st.sidebar.multiselect("Select Features", df.columns[:-1], default=df.columns[:-1])
    target_variable = "target"

    # Model Selection
    model_type = st.sidebar.selectbox("Select Model", ["Decision Tree", "Random Forest", "KNN", "SVM", "Gradient Boosting"])
    
    if model_type == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_type == "Random Forest":
        n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, 100)
        model = RandomForestClassifier(n_estimators=n_estimators)
    elif model_type == "KNN":
        n_neighbors = st.sidebar.slider("Number of Neighbors", 1, 20, 5)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model_type == "SVM":
        kernel = st.sidebar.selectbox("Kernel Type", ["linear", "rbf", "poly"])
        model = SVC(kernel=kernel)
    elif model_type == "Gradient Boosting":
        learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.1)
        model = GradientBoostingClassifier(learning_rate=learning_rate)

    # Train-Test Split
    X = df[selected_features]
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Performance Metrics
    st.write("### Model Performance")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.text(classification_report(y_test, y_pred, target_names=target_names if dataset_source == "Select UCI Dataset" else None))

    # Data Visualization
    st.write("### Data Visualization")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    if st.sidebar.checkbox("Show Histograms"):
        df.hist(figsize=(10, 5))
        st.pyplot(plt)
    if st.sidebar.checkbox("Show Violin Plots"):
        plt.figure(figsize=(10,5))
        sns.violinplot(data=df)
        st.pyplot(plt)

    # Live Prediction
    st.write("### Make Predictions")
    input_data = [st.slider(f"Enter {col}", float(df[col].min()), float(df[col].max()), float(df[col].mean())) for col in selected_features]
    if st.button("Predict"):
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)
        st.write(f"Predicted Outcome: {prediction[0]}")


st.markdown(
    """
    <div style="position: fixed; bottom: 10px; right: 10px; color: gray; font-size: 14px;">
        Created by Jayed Akhtar
    </div>
    """,
    unsafe_allow_html=True
)




