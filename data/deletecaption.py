import pandas as pd

# List of CSV file names
csv_files = ['test.csv', 'train.csv', 'val.csv']

# Loop through each file and remove the 'caption' column
for file in csv_files:
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Check if 'caption' column exists before dropping
    if 'caption_en' in df.columns:
        df = df.drop(columns=['caption_en'])
    
    # Save the updated CSV back to the file
    df.to_csv(file, index=False)

print("Columns 'caption' have been removed from the CSV files.")
