import logging

import pandas as pd
from src.data_cleaning import DataCleaning, DataPreprocessingStrategy


def get_data_for_test():
    try:
        df = pd.read_csv("C:\\Users\\This  PC\\Documents\\project mlops\\Data\\heart.csv")
        df = df.sample(n=100)
        preprocess_strategy = DataPreprocessingStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        df = data_cleaning.handle_data()
        df.drop(["HeartDisease"], axis=1, inplace=True)
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(e)
        raise e


 """
import pyodbc

def fetch_data_from_database_and_convert_to_csv():
    # Connect to the database
    conn = pyodbc.connect('Driver={SQL Server};'
                          'Server=server_name;'
                          'Database=database_name;'
                          'Trusted_Connection=yes;')

    # Fetch data from the database
    sql_query = pd.read_sql_query('SELECT * FROM table_name', conn)

    # Convert the data to a DataFrame
    df = pd.DataFrame(sql_query)

    # Write the data to a CSV file
    df.to_csv('output.csv', index=False)
"""
