import pandas as pd
from sas7bdat import SAS7BDAT

with SAS7BDAT('nhanes_17.sas7bdat') as file:
    # This file contains some necessary features that we need to do the model
    df_vita = file.to_data_frame()

# This file contains the response that we need to model (that is, whether the patient is told to have disease)
df_response = pd.read_excel("DIQ_J.xlsx")

# Data Cleaning process
# This is the procedure to get all the data we need


selected_vita_feature = df_vita[['SEQN', 'RIDAGEYR', 'smoking', 'alcohol', 'income', 'PA', 'BMXBMI', 'race', 'edu', 'menopause']]
selected_response = df_response[['SEQN', 'DIQ010']]

# Merging DataFrames on 'SEQN'
data_T2DM = pd.merge(selected_response, selected_vita_feature, on='SEQN', how='inner')
data_T2DM.columns = ["ID", "DM", "Age", "Smoking",'Alcohol', 'Income', 'PA', 'BMI', 'Race', 'Edu', 'Menopause']
# load the bp data as well
data_bp = pd.read_csv('bp.csv')
data_bp = data_bp.iloc[:,1:]
data_T2DM = pd.merge(data_T2DM, data_bp, on='ID', how='inner')
data_T2DM.to_csv("T2DM.csv", index=False)