import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# Data loading function
def load_customer_support_data(file_path):
    """
    Load and perform initial preprocessing of customer support dataset
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Basic data cleaning
    for col in ['message_input', 'message_output', 'final_message_sent']:
        if col in df.columns:
            # Replace NaN with empty string
            df[col] = df[col].fillna('')
            # Convert to string
            df[col] = df[col].astype(str)
    
    # Convert timestamps
    for col in ['created_at', 'synced_at']:
        if col in df.columns and df[col].dtype != 'datetime64[ns]':
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Add derived columns - handling UPPERCASE action values
    # Ensure sender_action is string and handle potential NaNs before comparison
    df['sender_action'] = df['sender_action'].fillna('').astype(str)
    df['suggestion_accepted'] = df['sender_action'] == 'ACCEPTED'
    df['suggestion_edited'] = df['sender_action'] == 'EDITED'
    df['suggestion_ignored'] = df['sender_action'] == 'IGNORED'
    df['suggestion_skipped'] = df['sender_action'] == 'SKIPPED'
    
    # Flag messages where suggestion wasn't used (for analysis)
    df['needs_analysis'] = (df['suggestion_ignored'] | df['suggestion_edited'] | df['suggestion_skipped'])
    
    # Create a column that shows how different the final message is from suggested
    # Initialize as float to avoid FutureWarning
    df['edit_distance'] = 0.0  # <-- FIX: Initialize as float
    mask = df['suggestion_edited'] & (df['message_output'] != '') & (df['final_message_sent'] != '')
    
    # Calculate edit distance (percentage difference in length)
    # Check if mask identifies any rows before applying
    if mask.any():
        # Ensure denominators are not zero before division
        denominator = df.loc[mask, 'message_output'].str.len().clip(lower=1) # Use Series.clip
        numerator = (df.loc[mask, 'message_output'].str.len() - df.loc[mask, 'final_message_sent'].str.len()).abs()
        df.loc[mask, 'edit_distance'] = (numerator / denominator) * 100.0

    print(f"Loaded {len(df)} records.")
    print(f"Actions breakdown:")
    print(df['sender_action'].value_counts(dropna=False)) # Keep dropna=False to see potential empty strings resulting from fillna
    
    return df

# Data exploration function
def explore_dataset(df):
    """
    Generate basic statistics about the dataset
    """
    # Ensure required columns exist before calculating stats
    stats = {}
    stats['total_messages'] = len(df)
    stats['unique_intents'] = df['detected_intent'].nunique() if 'detected_intent' in df.columns else 0
    stats['intent_distribution'] = df['detected_intent'].value_counts().to_dict() if 'detected_intent' in df.columns else {}
    stats['acceptance_rate'] = df['suggestion_accepted'].mean() * 100 if 'suggestion_accepted' in df.columns else 0.0
    stats['edit_rate'] = df['suggestion_edited'].mean() * 100 if 'suggestion_edited' in df.columns else 0.0
    stats['ignore_rate'] = df['suggestion_ignored'].mean() * 100 if 'suggestion_ignored' in df.columns else 0.0
    stats['skip_rate'] = df['suggestion_skipped'].mean() * 100 if 'suggestion_skipped' in df.columns else 0.0
    stats['unique_agents'] = df['sender_id'].nunique() if 'sender_id' in df.columns else 0
    stats['avg_message_length'] = df['message_input'].str.len().mean() if 'message_input' in df.columns else 0.0
    stats['avg_suggestion_length'] = df['message_output'].str.len().mean() if 'message_output' in df.columns else 0.0
    stats['avg_final_length'] = df['final_message_sent'].str.len().mean() if 'final_message_sent' in df.columns else 0.0

    # Sample messages from each action category for qualitative review
    samples = {}
    # Ensure sender_action exists before sampling
    if 'sender_action' in df.columns:
        for action in ['ACCEPTED', 'EDITED', 'IGNORED', 'SKIPPED']:
            action_df = df[df['sender_action'] == action]
            if len(action_df) > 0:
                sample_size = min(5, len(action_df))
                sample_df = action_df.sample(sample_size, random_state=42) # Add random state for reproducibility
                samples[action] = []
                # Ensure columns exist before accessing in sample loop
                required_sample_cols = ['message_input', 'message_output', 'final_message_sent', 'detected_intent']
                present_cols = {col for col in required_sample_cols if col in sample_df.columns}

                for _, row in sample_df.iterrows():
                    sample_entry = {
                        'input': row['message_input'] if 'message_input' in present_cols else 'N/A',
                        'suggestion': row['message_output'] if 'message_output' in present_cols else 'N/A',
                        'final': row['final_message_sent'] if 'final_message_sent' in present_cols else 'N/A',
                        'intent': row['detected_intent'] if 'detected_intent' in present_cols else 'N/A'
                    }
                    samples[action].append(sample_entry)
            else:
                 samples[action] = [] # Add empty list if no samples found for the action

    return stats, samples

# Create analysis dataset
def create_analysis_dataset(df):
    """
    Create a focused dataset for analysis, containing only cases that need investigation
    """
    print("Creating analysis datasets...")
    
    # Make sure we have strings, not NaN values (redundant if load_customer_support_data handles this)
    for col in ['message_input', 'message_output', 'final_message_sent', 'sender_action']:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)

    # Ensure required boolean flag columns exist
    required_flags = ['suggestion_accepted', 'suggestion_ignored', 'suggestion_edited', 'suggestion_skipped']
    for flag in required_flags:
         if flag not in df.columns:
              print(f"Warning: Required flag column '{flag}' missing. Creating default (False).")
              df[flag] = False # Add a default column if missing

    # Create analysis dataset - non-accepted records with valid input/output
    has_input = df['message_input'] != '' if 'message_input' in df.columns else pd.Series([True] * len(df))
    has_output = df['message_output'] != '' if 'message_output' in df.columns else pd.Series([True] * len(df))
    # sender_action is already string and notna checked in load_customer_support_data
    is_not_accepted = ~df['suggestion_accepted']
    
    # Filter for analysis dataset
    analysis_df = df[has_input & has_output & is_not_accepted].copy()
    
    # For cases where suggestion was rejected but a final message was sent
    has_final = df['final_message_sent'] != '' if 'final_message_sent' in df.columns else pd.Series([True] * len(df))
    comparison_df = df[has_input & has_output & has_final & is_not_accepted].copy()
    
    # Add metadata useful for analysis
    if not comparison_df.empty:
        if 'message_input' in comparison_df.columns:
            comparison_df['message_length'] = comparison_df['message_input'].str.len()
        if 'message_output' in comparison_df.columns:
            comparison_df['suggestion_length'] = comparison_df['message_output'].str.len()
        if 'final_message_sent' in comparison_df.columns:
            comparison_df['final_length'] = comparison_df['final_message_sent'].str.len()
    
    print(f"Analysis dataset: {len(analysis_df)} records")
    print(f"Comparison dataset: {len(comparison_df)} records")
    
    return analysis_df, comparison_df

# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "data-new.csv" # Make sure this file exists
    output_dir = "./output_test_preprocess" # Define separate output dir for testing
    analysis_output_path = os.path.join(output_dir, "analysis_dataset.csv")
    comparison_output_path = os.path.join(output_dir, "comparison_dataset.csv")

    os.makedirs(output_dir, exist_ok=True) # Create test output dir

    if os.path.exists(file_path):
        # Load and process data
        df = load_customer_support_data(file_path)

        # Get statistics and samples
        stats, samples = explore_dataset(df)
        print("\nDataset Statistics:")
        for key, value in stats.items():
            if not isinstance(value, dict):
                print(f"- {key}: {value:.2f}" if isinstance(value, (int, float)) else f"- {key}: {value}")

        # Create analysis datasets
        analysis_df, comparison_df = create_analysis_dataset(df)

        # Save processed datasets for next steps
        try:
             analysis_df.to_csv(analysis_output_path, index=False)
             comparison_df.to_csv(comparison_output_path, index=False)
             print(f"Processed files saved to {output_dir}")
        except Exception as e:
             print(f"Error saving processed files: {e}")
    else:
        print(f"Error: Input file not found at {file_path}")