import pandas as pd

data = pd.read_csv('../combined_data.csv')[['Date', 'Total_Current_Positive_Cases', 'Recovered']]
early_data = data[data['Date'] < '2020-04-01']
# Merge Columns by Date and summing the values in the other columns
early_data = early_data.groupby('Date').sum().reset_index()
print(early_data)