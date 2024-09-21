import pandas as pd
import csv

# Read the CSV file into a DataFrame
input_name = "Location.csv"
output_name = "filtered_Locations.csv"
df = pd.read_csv(input_name)

# Filter the DataFrame based on the 'checkstat' column
filtered_df = df[df['checkstat'] == 'MARKER_GESETZT']

# Write the filtered DataFrame to a new CSV file
filtered_df.to_csv(output_name , index=False, quoting=csv.QUOTE_ALL) #quote all to not mess up with , symbols in strings

print("Filtered CSV file has been created successfully.")
