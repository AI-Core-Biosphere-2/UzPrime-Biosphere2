import pandas as pd
import glob

def load_and_merge_csvs(folder_path: str) -> pd.DataFrame:
    """
    Load all CSV files from a folder, preprocess them, and merge into a single DataFrame.
    Handles files with different columns using an outer join.
    """
    csv_files = glob.glob(f"{folder_path}/*.csv")
    if not csv_files:
        raise ValueError("No CSV files found in the provided folder path.")
    
    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])
            elif 'DateTime' in df.columns:
                df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
                df = df.dropna(subset=['DateTime'])
            else:
                print(f"Warning: {file} does not contain a time column and will be skipped.")
                continue
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not df_list:
        raise ValueError("No valid CSV files were loaded.")
    
    # Merge using an outer join to preserve all columns from different files
    merged_df = pd.concat(df_list, join='outer', ignore_index=True)
    return merged_df

if __name__ == "__main__":
    folder_path = "path/to/your/starter_data_folder"  # Update this path accordingly
    merged_df = load_and_merge_csvs(folder_path)
    print("Merged Data Preview:")
    print(merged_df.head())
