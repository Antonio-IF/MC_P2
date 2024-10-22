
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: data.py : python script for data collection                                                 -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                               -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import pandas as pd

# Define a class to handle the process of loading, filtering, and saving the dataset.
class DataProcessor:
    def __init__(self, file_path):
        # Constructor that loads the dataset
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
    
    def filter_loan_type(self, loan_type):
        # Filter the dataset to include only the specified loan type
        self.data = self.data[self.data['Type_of_Loan'].str.contains(loan_type, na=False, case=False)]

    def save_to_excel(self, output_file_path):
        # Save the filtered dataset as an Excel file
        self.data.to_excel(output_file_path, index=False)
        return output_file_path

# Create an instance of the class and process the data
processor = DataProcessor('Data/train-2.csv')
processor.filter_loan_type("Personal Loan")
output_file_path = 'Data/train_filtered_personal_loan.xlsx'
processor.save_to_excel(output_file_path)

