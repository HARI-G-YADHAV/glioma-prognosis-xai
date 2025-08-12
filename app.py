import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the pre-trained SVM model
with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

# Streamlit interface
def main():
    st.title("Patient Classification using SVM")
    st.write("Upload a CSV file with patient details to predict their class.")

    # Upload file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file)

        # Display the uploaded data
        st.write("Uploaded Data:")
        st.write(data.head())

        # Updated required features based on your provided list
        required_features = [
            'Age_at_diagnosis', 'IDH1', 'TP53', 'ATRX', 'PTEN', 'EGFR', 'CIC',
            'MUC16', 'PIK3CA', 'NF1', 'PIK3R1', 'FUBP1', 'RB1', 'NOTCH1', 'BCOR',
            'CSMD3', 'SMARCA4', 'GRIN2A', 'IDH2', 'FAT4', 'PDGFRA'
        ]

        # Add missing columns to the uploaded data, initializing them with 0
        for feature in required_features:
            if feature not in data.columns:
                data[feature] = 0

        # Now that we have all the required columns, we can proceed
        X = data[required_features]
        
        # Scale the numerical features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Make predictions using the SVM model
        predictions = svm_model.predict(X_scaled)

        # Add predictions to the dataframe
        data['Prediction'] = predictions

        # Display predictions
        st.write("Predictions:")
        st.write(data[['Prediction']])

        # Option to download the data with predictions
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name='patient_predictions.csv',
            mime='text/csv'
        )
        
if __name__ == "__main__":
    main()
