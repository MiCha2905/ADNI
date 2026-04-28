import pandas as pd
import os

def prepare_data():
    # 1. Load the FastSurfer features
    features_path = "fastsurfer_features.csv"
    if not os.path.exists(features_path):
        print(f"Error: {features_path} not found.")
        return
    
    print("Loading FastSurfer features...")
    features = pd.read_csv(features_path)
    
    # 2. Extract subject_id and scan_date from the 'subject' column
    # Format example: '002_S_0413_2007-06-01_07_57_43.0'
    # We split by '_' and take the first 3 parts for subject_id, and 4th part for date.
    
    # Extract subject_id (e.g., '002_S_0413')
    features['subject_id'] = features['subject'].apply(lambda x: "_".join(x.split('_')[:3]))
    
    # Extract scan_date (e.g., '2007-06-01')
    features['scan_date'] = features['subject'].apply(lambda x: x.split('_')[3])
    # Convert to proper datetime objects
    features['scan_date'] = pd.to_datetime(features['scan_date'])
    
    print(f"Extracted {features['subject_id'].nunique()} unique subjects from features.")
    
    # 3. Load your clinical labels
    labels_path = "ALL_4_27_2026.csv" 
    
    if not os.path.exists(labels_path):
        print(f"\n[!] Waiting for the labels file: {labels_path}")
        return
        
    print("Loading clinical labels...")
    df_final = pd.read_csv(labels_path)
    
    # Rename 'Subject' to 'subject_id' to match the features dataframe
    if 'Subject' in df_final.columns:
        df_final = df_final.rename(columns={'Subject': 'subject_id'})
        
    # Since subjects have multiple visits in the labels file (m12, m24, bl),
    # we just need the diagnosis (Group). We'll drop duplicates to get a unique mapping.
    df_final_unique = df_final[['subject_id', 'Group']].drop_duplicates(subset=['subject_id'])
        
    # 4. Merge features with labels
    print("Merging features with clinical labels...")
    df = features.merge(df_final_unique, on="subject_id")

    # 5. Output results
    print("\n--- Merge Complete ---")
    print("Final shape:", df.shape)
    print("\nClass distribution:\n", df["Group"].value_counts())
    
    # Save the merged dataset
    output_path = "merged_features_labels.csv"
    df.to_csv(output_path, index=False)
    print(f"\nMerged data saved to {output_path}")
    
    return df

if __name__ == "__main__":
    prepare_data()
