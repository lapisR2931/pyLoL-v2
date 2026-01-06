import csv
import os
import pandas as pd

def save_to_csv(data, filename, column_name):
    """Function to save data to CSV file"""
    file_exists = os.path.exists(filename)
    mode = 'a' if file_exists else 'w'
    
    with open(filename, mode=mode, newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([column_name])
        for value in data:
            writer.writerow([value])

