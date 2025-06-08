import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Load the original data to get unique values for categorical features and to refit scaler and encoder
file_path = '/content/drive/MyDrive/Phú Quốc/du_lieu_da_them_season_va_income( dữ liệu cuối cùng).xlsx - Sheet1.csv'
df = pd.read_csv(file_path)

# --- Data Preprocessing steps replicated from the training notebook ---
# Separate features (X) and target variable (y) for refitting
X_for_preprocessing = df.drop(columns=['product_detail', 'transaction_id', 'customer_name'])
y_for_preprocessing = df[['product_detail']]

# Convert 'transaction_date' to numerical (timestamp)
X_for_preprocessing['transaction_date'] = pd.to_datetime(X_for_preprocessing['transaction_date'], errors='coerce').astype(int) / 10**9

# Identify categorical columns (excluding 'transaction_date')
categorical_cols = X_for_preprocessing.select_dtypes(include=['object', 'category']).columns

# Apply one-hot encoding to categorical features to fit the encoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # Use handle_unknown='ignore' for new categories
encoder.fit(X_for_preprocessing[categorical_cols])


# Identify numerical columns (including 'transaction_date')
numerical_cols = ['transaction_qty', 'unit_price', 'Total_Bill', 'Age', 'Income', 'transaction_date']

# Apply standardization to numerical features to fit the scaler
scaler = StandardScaler()
scaler.fit(X_for_preprocessing[numerical_cols])

# Get the list of product details for inverse transformation
product_detail_list = df['product_detail'].unique().tolist()

# --- Streamlit App UI ---
st.title('Product Detail Prediction App')

st.write('Enter the details below to predict the product detail.')

# Add input fields for features
input_data = {}

# Numerical inputs
for col in numerical_cols:
    if col == 'transaction_date':
        date_value = st.date_input(f'Enter {col}')
        # Handle NaT from date_input if no date is selected initially
        if pd.isna(date_value):
             input_data[col] = None # Or a default value/mean
        else:
             input_data[col] = pd.to_datetime(date_value).timestamp()
    elif col in ['transaction_qty', 'Age', 'Income']:
         input_data[col] = st.number_input(f'Enter {col}', value=float(df[col].mean())) # Use mean as default, ensure float
    else:
        input_data[col] = st.number_input(f'Enter {col}', value=float(df[col].mean())) # Use mean as default, ensure float


# Categorical inputs
for col in categorical_cols:
    unique_values = df[col].unique().tolist()
    input_data[col] = st.selectbox(f'Select {col}', unique_values)

# Create a DataFrame from the input data
# Ensure column order matches training data by reindexing
input_df = pd.DataFrame([input_data])


# --- Prediction Logic ---
if st.button('Predict'):
    # Preprocess the input data
    input_df_processed = input_df.copy()

    # Apply standardization to numerical features
    input_df_processed[numerical_cols] = scaler.transform(input_df_processed[numerical_cols])

    # Apply one-hot encoding to categorical features
    input_categorical_encoded = encoder.transform(input_df_processed[categorical_cols])
    input_categorical_encoded_df = pd.DataFrame(input_categorical_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=input_df_processed.index)

    # Drop original categorical columns and concatenate with encoded ones
    input_df_processed = input_df_processed.drop(columns=categorical_cols)
    input_df_processed = pd.concat([input_df_processed, input_categorical_encoded_df], axis=1)

    # Ensure the columns in the input data match the columns in the training data
    # Add missing columns with a value of 0 (for one-hot encoded features not present in input)
    # This assumes the training data columns are the reference. You might need to load X_train columns if necessary.
    training_columns = X.columns # Assuming X from the previous notebook state holds the columns
    missing_cols = set(training_columns) - set(input_df_processed.columns)
    for c in missing_cols:
        input_df_processed[c] = 0
    # Ensure the order of columns is the same as in the training data
    input_df_processed = input_df_processed[training_columns]


    # Make prediction
    prediction_encoded = model.predict(input_df_processed)

    # Decode the prediction
    # Find the index of the predicted class (assuming multi-label where only one is predicted as 1)
    predicted_class_indices = np.where(prediction_encoded[0] == 1)[0]

    if len(predicted_class_indices) > 0:
        # Get the name of the predicted product detail from the encoder's feature names
        predicted_product_detail = encoder.get_feature_names_out(['product_detail'])[predicted_class_indices[0]]
        # Remove the 'product_detail_' prefix
        predicted_product_detail = predicted_product_detail.replace('product_detail_', '')
        st.write('Predicted Product Detail:', predicted_product_detail)
    else:
        st.write('Could not predict product detail.')
