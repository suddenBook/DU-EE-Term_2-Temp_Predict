import pandas as pd

file_path = '2020WeatherData.xlsx'

final_dataframe = pd.DataFrame()


for month in range(1, 13):
    month_name = pd.to_datetime('2020-' + str(month), format='%Y-%m').strftime('%B')
    monthly_data = pd.read_excel(file_path, sheet_name=month_name)
    
    monthly_data = monthly_data[~monthly_data['Day Number'].isin(['Results', 'Max/Min'])]
    
    selected_columns = monthly_data[['Day Number', 'Total rainfall', 'Average Dry Bulb Temp', 'Max Temp', 'Min Temp']]
    selected_columns.columns = ['Day', 'PPT.', 'Av temp', 'Tmax', 'Tmin']
    
    selected_columns.insert(0, 'Year', 2020)
    selected_columns.insert(1, 'Month', month)
    
    selected_columns['Date'] = pd.to_datetime(selected_columns[['Year', 'Month', 'Day']])
    
    final_dataframe = pd.concat([final_dataframe, selected_columns], ignore_index=True)

final_dataframe.to_csv('durhamtemp_2020.csv', index=False)