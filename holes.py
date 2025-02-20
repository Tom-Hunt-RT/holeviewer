import pandas as pd
import streamlit as st
import plotly.express as px

def loaddata():
    st.write("### Load Data")
    uploaded_file = st.file_uploader("Choose a file")
    
    if uploaded_file is not None:
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'iso-8859-1']
        
        for encoding in encodings_to_try:
            try:
                # Read the CSV using pandas while specifying the encoding
                uploaded_file.seek(0)  # Reset the file pointer
                drillhole_db = pd.read_csv(uploaded_file, encoding=encoding)
                return drillhole_db  # Return if the file is read successfully
            except UnicodeDecodeError:
                continue  # Try next encoding if one fails
        
        # If all encodings fail, show an error
        st.error("Unable to read the file with the tested encodings. Please check the file encoding.")
        return pd.DataFrame()
    
    else:
        st.warning("Please upload a file.")
        return pd.DataFrame()
