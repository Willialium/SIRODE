import os
import pandas as pd

def load_csv_files(directory_path):
    # Get a list of all CSV files in the directory
    csv_files = [file for file in os.listdir(directory_path) if file.endswith(".csv")]

    # Initialize an empty list to store DataFrames
    dfs = []

    # Loop through each CSV file and load it into a DataFrame
    for file in csv_files:
        file_path = os.path.join(directory_path, file)
        df = pd.read_csv(file_path)
        # convert data column to datetime
        df['data'] = pd.to_datetime(df['data'])
        # remove the time from the datetime
        df['data'] = df['data'].dt.date
        df.rename(columns={
            'data': 'Date',
            'stato':'Country',
            'codice_regione':'Region_Code',
            'denominazione_regione':'Region_Name',
            'ricoverati_con_sintomi':'Hospitalized_With_Symptoms',
            'terapia_intensiva':'Intensive_Care',
            'totale_ospedalizzati':'Total_Hospitalized',
            'isolamento_domiciliare':'Home_Confined',
            'totale_positivi':'Total_Current_Positive_Cases',
            'variazione_totale_positivi':'New_Positive_Cases',
            'nuovi_positivi':'New_Cases',
            'dimessi_guariti':'Recovered',
            'deceduti':'Dead',
            'casi_da_sospetto_diagnostico':'Positive_From_Clinical_Setting',
            'casi_da_screening':'Positive_From_State_Tests',
            'totale_casi':'Total_Positive_Cases',
            'tamponi':'Tests_Performed',
            'tamponi_test_molecolare':'Molecular_Tests_Performed',
            'tamponi_test_antigenico_rapido':'Antigen_Tests_Performed',
            'totale_positivi_test_molecolare':'Total_Positive_Molecular_Tests',
            'totale_positivi_test_antigenico_rapido':'Total_Positive_Antigen_Tests',
            'casi_testati':'People_Tested',
            'note':'Notes',
            'ingressi_terapia_intensiva':'Daily_Intensive_Care',
            'note_test':'Test_Notes',
            'note_casi':'Case_Notes'}, inplace=True)

        # Drop columns that are not needed

        dfs.append(df)

    # Concatenate all DataFrames in the list into one
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.drop(columns=['Country', 'Region_Code', 'lat', 'long', 'Notes', 'Test_Notes', 'Case_Notes', 'codice_nuts_1', 'codice_nuts_2'], inplace=True)

    return combined_df

def save_combined_dataframe(dataframe, output_filename="combined_data.csv"):
    # Save the combined DataFrame to a CSV file in the local directory
    output_path = os.path.join(os.getcwd(), output_filename)
    dataframe.to_csv(output_path, index=False)
    print(f"Combined data saved to {output_path}")

# Replace 'your_directory_path' with the path to your directory containing CSV files
directory_path = 'Italy_data/dati-regioni'

# Load CSV files into a DataFrame
combined_data = load_csv_files(directory_path)

# Save the combined DataFrame to a CSV file in the local directory
save_combined_dataframe(combined_data)
