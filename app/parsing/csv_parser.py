import pandas as pd

from app.parsing.base_parser import BaseParser

# Create a new parser class that inherits from BaseParser and implements the parse method to handle CSV files. The parse method reads the CSV file using pandas, iterates through each row, and constructs a string representation of the data in a readable format. Finally, it joins all the rows into a single string and returns it.
class CSVParser(BaseParser):

    def parse(self, file_path):
        """
        Parse CSV file from path and return raw text
        """

        # Load CSV into panda DataFrame to transform into natural language
        df = pd.read_csv(file_path)

        # Processed text stored in rows list
        # "Name: Alice, Major: CS, Year: Senior"
        rows = []

        # Loop through each row in the DataFrame and construct a string representation of the data
        for _, row in df.iterrows():
            row_text = []

            # Loop through each column in the row and create a string in the format "ColumnName: Value"
            for column in df.columns:

                # Join rows together 
                row_text.append(f"{column}: {row[column]}")

            rows.append(", ".join(row_text))

        # return raw text with each row separated by a newline character
        return "\n".join(rows)