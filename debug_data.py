import pandas as pd

# Load the data
df = pd.read_csv('data-new.csv')
print(f"Total records: {len(df)}")

# Check columns
print("\nColumns in dataset:")
print(df.columns.tolist())

# Check action types
print("\nAction breakdown:")
print(df['sender_action'].value_counts(dropna=False))

# Convert to uppercase for testing
if 'sender_action' in df.columns and df['sender_action'].dtype == 'object':
    df['sender_action_upper'] = df['sender_action'].str.upper()
    
# Create test flags
df['test_accepted'] = df['sender_action_upper'] == 'ACCEPTED'
df['test_edited'] = df['sender_action_upper'] == 'EDITED'
df['test_ignored'] = df['sender_action_upper'] == 'IGNORED'
df['test_skipped'] = df['sender_action_upper'] == 'SKIPPED'

# Check message fields
print("\nMessage fields presence:")
print(f"Non-empty message_input: {df['message_input'].notna().sum()}")
print(f"Non-empty message_output: {df['message_output'].notna().sum()}")
print(f"Non-empty final_message_sent: {df['final_message_sent'].notna().sum()}")

# Sample of potential analysis records
print("\nSample of potential analysis records:")
sample_df = df[(df['test_edited'] | df['test_ignored'] | df['test_skipped'])].head(5)
if len(sample_df) > 0:
    for _, row in sample_df.iterrows():
        print(f"\nID: {row.get('id', 'N/A')}")
        print(f"Action: {row.get('sender_action', 'N/A')}")
        print(f"Input: {row.get('message_input', 'N/A')[:50]}...")
        print(f"Output: {row.get('message_output', 'N/A')[:50]}...")
        print(f"Final: {row.get('final_message_sent', 'N/A')[:50]}...")
else:
    print("No sample records found!")

print("\nDebug complete!")