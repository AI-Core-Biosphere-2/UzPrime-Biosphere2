import pandas as pd
import glob

def load_and_merge_csvs(folder_path: str) -> pd.DataFrame:
    """
    Load all CSV files from a folder, preprocess them, and merge into a single DataFrame.
    This function uses an outer join to preserve all columns from different files.
    """
    csv_files = glob.glob(f"{folder_path}/*.csv")
    if not csv_files:
        raise ValueError("No CSV files found in the provided folder path.")
    
    df_list = []
    for file in csv_files:
        try:
            # Read CSV with low_memory set to False to avoid dtype warnings
            df = pd.read_csv(file, low_memory=False)
            # Check for a time column, using either 'DateTime' or 'timestamp'
            if 'DateTime' in df.columns:
                time_col = 'DateTime'
            elif 'timestamp' in df.columns:
                time_col = 'timestamp'
            else:
                print(f"Warning: {file} does not contain a recognizable time column.")
                time_col = None
            
            if time_col:
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                df = df.dropna(subset=[time_col])
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not df_list:
        raise ValueError("No valid CSV files were loaded.")
    
    # Merge using an outer join to keep all columns from all files
    merged_df = pd.concat(df_list, join='outer', ignore_index=True)
    return merged_df

if __name__ == "__main__":
    folder_path = "/Users/sheraliozodov/UzPrime-Biosphere2/data"  # Update this path accordingly
    merged_df = load_and_merge_csvs(folder_path)
    print("Merged Data Preview:")
    print(merged_df.head())
