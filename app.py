import streamlit as st
import pickle
import pandas as pd

# Load the model
with open("C:/Users/User/Documents/Machine Learning Projects/credit_card_logistic_regression_model.pkl", 'rb') as f:
    clf = pickle.load(f)

# Load the encoders
with open('C:/Users/User/Documents/Machine Learning Projects/OH_encoder.pkl', 'rb') as f:
    OH_encoder = pickle.load(f)



st.title("Chun Yew's Project - Credit Card Approval Prediction")
st.image("""https://wallpapers.com/images/hd/stack-of-dark-colored-credit-cards-8kh9jkmkkurlact3.jpg""")
st.header('Enter the characteristics of the applicants:')

# Create a dictionary to hold user input
user_input = {}
user_input['Age'] = st.number_input('Age:', min_value=15.0, max_value=100.0, value=15.0)
user_input['Debt'] = st.number_input('Debt Amount:', min_value=0.0, max_value=20.0, value=1.0)
user_input['Married'] = st.selectbox('Married:', ['Yes', 'No'])
user_input['BankCustomer'] = st.selectbox('Bank Customer:', ['Yes', 'No'])
user_input['Industry'] = st.selectbox('Industry:', ['Industrials', 'Materials', 'CommunicationServices', 'Transport',
 'InformationTechnology','Financials', 'Energy', 'Real Estate', 'Utilities',
 'ConsumerDiscretionary', 'Education', 'ConsumerStaples', 'Healthcare',
 'Research'])
user_input['Ethnicity'] = st.selectbox('Ethnicity:', ['White', 'Black', 'Asian', 'Latino', 'Other'])
user_input['YearsEmployed'] = st.number_input('Years Employed:', min_value=0.0, max_value=10.0, value=1.0)
user_input['PriorDefault'] = st.selectbox('Prior Default:', ['Yes', 'No'])
user_input['Employed'] = st.selectbox('Employed:', ['Yes', 'No'])
user_input['CreditScore'] = st.number_input('Credit Score:', min_value=0.0, max_value=10.0, value=1.0)
user_input['Citizen'] = st.selectbox('Citizen:', ['ByBirth', 'ByOtherMeans' ,'Temporary'])
user_input['ZipCode'] = st.number_input('Zip Code:', min_value=0.0, max_value=99999.0)
user_input['Income'] = st.number_input('Income:', min_value=1.0, max_value=100000.0, value=10.0)

# Convert 'Yes'/'No' to 1/0
user_input['Married'] = 1 if user_input['Married'] == 'Yes' else 0
user_input['BankCustomer'] = 1 if user_input['BankCustomer'] == 'Yes' else 0
user_input['PriorDefault'] = 1 if user_input['PriorDefault'] == 'Yes' else 0
user_input['Employed'] = 1 if user_input['Employed'] == 'Yes' else 0

# Convert the user input into a DataFrame
input_data = pd.DataFrame([user_input])

# Select object (categorical) columns
s = (input_data.dtypes == 'object')
object_cols = list(s[s].index)

# One-hot encode categorical features
OH_cols_new = pd.DataFrame(OH_encoder.transform(input_data[object_cols]))
OH_cols_new.index = input_data.index
OH_cols_new.columns = OH_encoder.get_feature_names_out(object_cols)

input_data = input_data.drop(object_cols, axis=1)
input_data = pd.concat([input_data, OH_cols_new], axis=1)

if st.button('Predict Approval'):
    Approval = clf.predict(input_data)
    if Approval[0] == 1:
        st.success('The credit card is predicted to be approved.')
    else:
        st.error('The credit card is predicted to be rejected.')
