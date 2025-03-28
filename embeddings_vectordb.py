import pandas as pd
import numpy as np
from tqdm import tqdm
import os
# import json # No longer needed here
from sentence_transformers import SentenceTransformer
import chromadb
# from chromadb.utils import embedding_functions # Not needed if providing embeddings directly

# --- Configuration Constants ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Or 'paraphrase-multilingual-MiniLM-L12-v2' etc.
# --- End Configuration Constants ---


# Initialize the embedding model
def init_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer: # Added type hints
    """Initialize the sentence transformer model for generating embeddings."""
    print(f"Loading embedding model: {model_name}...")
    # Using a lightweight model suitable for semantic similarity
    try:
        model = SentenceTransformer(model_name)
        print("Embedding model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading embedding model '{model_name}': {e}")
        raise # Re-raise the exception to halt execution if model loading fails


# Removed generate_embeddings function as model.encode is used directly


# Set up ChromaDB for vector storage
def setup_chroma_db(db_path: str = "./chroma_db") -> tuple[chromadb.PersistentClient, dict[str, chromadb.Collection]]: # Added type hints
    """Initialize ChromaDB client and get/create collections."""
    print(f"Setting up ChromaDB at {db_path}...")

    # Create directory if it doesn't exist
    os.makedirs(db_path, exist_ok=True)

    # Initialize client
    try:
        client = chromadb.PersistentClient(path=db_path)
        print("ChromaDB client initialized.")
    except Exception as e:
        print(f"Error initializing ChromaDB client at '{db_path}': {e}")
        raise

    # Define collection names
    collection_names = {
        "customer_messages": "Customer support inquiries",
        "suggested_responses": "AI-generated suggested responses",
        "final_responses": "Actual responses sent to customers",
        "rejected_pairs": "Pairs of rejected suggestions and their replacements"
    }

    collections = {}
    for name, desc in collection_names.items():
        try:
            # Get or create collection - no need to specify embedding function if passing embeddings directly
            collections[name] = client.get_or_create_collection(
                name=name,
                # embedding_function=ef, # Not needed if passing embeddings during upsert/add
                metadata={"description": desc, "hnsw:space": "cosine"} # Specify distance metric if desired
            )
            print(f"Collection '{name}' accessed/created.")
        except Exception as e:
            # Handle potential errors during collection creation/access
            # Log the error, potentially skip this collection or raise an error
            print(f"Error getting/creating collection '{name}': {e}")
            # Depending on requirements, you might want to raise e here
            # Or continue and handle the missing collection in store_in_chroma

    # Check if all expected collections were created/accessed
    if len(collections) != len(collection_names):
         print("Warning: Not all expected ChromaDB collections could be accessed or created.")
         # Potentially raise an error if certain collections are critical

    return client, collections


# Store data in ChromaDB
def store_in_chroma(df: pd.DataFrame, collections: dict[str, chromadb.Collection], model: SentenceTransformer, batch_size: int = 100): # Added type hints
    """Store preprocessed data in ChromaDB collections using upsert."""
    print("Storing data in vector database using upsert...")

    # Ensure the DataFrame is not empty
    if df.empty:
        print("Input DataFrame is empty. Nothing to store.")
        return

    total_batches = (len(df) - 1) // batch_size + 1
    print(f"Processing {len(df)} records in {total_batches} batches of size {batch_size}.")

    # Process in batches for efficiency
    for i in tqdm(range(0, len(df), batch_size), desc="Storing Batches"):
        batch_df = df.iloc[i:i+batch_size].copy() # Use .copy() to avoid SettingWithCopyWarning

        # --- Prepare common data for the batch ---
        batch_ids = batch_df['id'].astype(str).tolist()
        common_metadata_list = []
        try:
            for _, row in batch_df.iterrows():
                 # Construct metadata safely using .get() with defaults
                 meta = {
                    "job_id": str(row.get('job_id', 'N/A')),
                    # Ensure intent is int, handle potential errors/NaN
                    "detected_intent": int(row['detected_intent']) if pd.notna(row.get('detected_intent')) else -1,
                    "sender_action": str(row.get('sender_action', 'N/A')),
                    # Ensure datetime is string, handle potential NaT
                    "created_at": str(row['created_at']) if pd.notna(row.get('created_at')) else 'N/A',
                    "sender_id": str(row.get('sender_id', 'N/A'))
                 }
                 common_metadata_list.append(meta)
        except Exception as meta_err:
             print(f"\nError preparing common metadata in batch starting at index {i}: {meta_err}")
             print(f"Problematic row data (first few columns): \n{batch_df.iloc[0, :5]}...") # Print first row's data
             continue # Skip this batch if metadata preparation fails

        # --- 1. Process Customer Messages ---
        if "customer_messages" in collections and 'message_input' in batch_df.columns:
            customer_texts = batch_df['message_input'].tolist()
            try:
                customer_embeddings = model.encode(customer_texts, show_progress_bar=False).tolist() # Encode once
                collections["customer_messages"].upsert( # <-- Use upsert
                    ids=batch_ids,
                    embeddings=customer_embeddings,
                    documents=customer_texts,
                    metadatas=common_metadata_list
                )
            except Exception as e:
                print(f"\nError upserting customer_messages batch starting at index {i}: {e}")
                # Optionally add more details, e.g., print batch_ids

        # --- 2. Process Suggested Responses ---
        if "suggested_responses" in collections and 'message_output' in batch_df.columns:
            suggestion_ids = [f"sugg_{id}" for id in batch_ids]
            suggestion_texts = batch_df['message_output'].tolist()
            try:
                suggestion_embeddings = model.encode(suggestion_texts, show_progress_bar=False).tolist() # Encode once
                collections["suggested_responses"].upsert( # <-- Use upsert
                    ids=suggestion_ids,
                    embeddings=suggestion_embeddings,
                    documents=suggestion_texts,
                    metadatas=common_metadata_list # Use the same common metadata
                )
            except Exception as e:
                 print(f"\nError upserting suggested_responses batch starting at index {i}: {e}")


        # --- 3. Process Final Responses (where they exist) ---
        final_df = batch_df[batch_df['final_message_sent'].fillna('') != ''].copy()
        if "final_responses" in collections and not final_df.empty:
            final_ids = [f"final_{id}" for id in final_df['id'].astype(str).tolist()]
            final_texts = final_df['final_message_sent'].tolist()
            # Prepare metadata specific to final_df rows
            final_metadata_list = []
            try:
                 for _, row in final_df.iterrows():
                     meta = {
                         "job_id": str(row.get('job_id', 'N/A')),
                         "detected_intent": int(row['detected_intent']) if pd.notna(row.get('detected_intent')) else -1,
                         "sender_action": str(row.get('sender_action', 'N/A')),
                         "created_at": str(row['created_at']) if pd.notna(row.get('created_at')) else 'N/A',
                         "sender_id": str(row.get('sender_id', 'N/A'))
                     }
                     final_metadata_list.append(meta)

                 final_embeddings = model.encode(final_texts, show_progress_bar=False).tolist() # Encode once
                 collections["final_responses"].upsert( # <-- Use upsert
                    ids=final_ids,
                    embeddings=final_embeddings,
                    documents=final_texts,
                    metadatas=final_metadata_list
                 )
            except Exception as e:
                 print(f"\nError upserting final_responses batch starting at index {i}: {e}")


        # --- 4. Process Rejection Pairs ---
        # Filter based on sender_action (already string) and non-empty final message
        rejected_df = batch_df[
            batch_df['sender_action'].isin(['EDITED', 'IGNORED']) & # Case-sensitive match
            (batch_df['final_message_sent'].fillna('') != '')
        ].copy()

        if "rejected_pairs" in collections and not rejected_df.empty:
            pair_ids = [f"pair_{id}" for id in rejected_df['id'].astype(str).tolist()]
            # Combine texts for document storage
            pair_texts = [
                f"CUSTOMER: {row.get('message_input', '')}\nSUGGESTION: {row.get('message_output', '')}\nFINAL: {row.get('final_message_sent', '')}"
                for _, row in rejected_df.iterrows()
            ]
            # Create text used for embedding (e.g., suggestion + final)
            embedding_texts = [
                f"{row.get('message_output', '')} {row.get('final_message_sent', '')}"
                for _, row in rejected_df.iterrows()
            ]
            # Prepare metadata specific to rejected_df rows
            pair_metadata_list = []
            try:
                 for _, row in rejected_df.iterrows():
                     meta = {
                        "job_id": str(row.get('job_id', 'N/A')),
                        "detected_intent": int(row['detected_intent']) if pd.notna(row.get('detected_intent')) else -1,
                        "sender_action": str(row.get('sender_action', 'N/A')),
                        # Ensure edit_distance is float, handle potential errors/NaN
                        "edit_distance": float(row['edit_distance']) if pd.notna(row.get('edit_distance')) else 0.0,
                        "sender_id": str(row.get('sender_id', 'N/A'))
                     }
                     pair_metadata_list.append(meta)

                 pair_embeddings = model.encode(embedding_texts, show_progress_bar=False).tolist() # Encode once
                 collections["rejected_pairs"].upsert( # <-- Use upsert
                    ids=pair_ids,
                    embeddings=pair_embeddings,
                    documents=pair_texts,
                    metadatas=pair_metadata_list
                 )
            except Exception as e:
                 print(f"\nError upserting rejected_pairs batch starting at index {i}: {e}")

    print("Finished storing data.")


# Example usage
if __name__ == "__main__":
    # Configuration for standalone run
    input_file_path = "./output/comparison_dataset.csv" # Assumes comparison data exists from preprocess step
    output_db_path = "./chroma_db_test_embed" # Use a separate DB path for testing

    print(f"Running embeddings_vectordb.py as standalone script.")
    print(f"Input comparison data: {input_file_path}")
    print(f"Output ChromaDB path: {output_db_path}")

    # --- Optional: Clean up previous test run ---
    # import shutil
    # if os.path.exists(output_db_path):
    #     print(f"Deleting existing test database at {output_db_path}...")
    #     shutil.rmtree(output_db_path)
    # ---

    if os.path.exists(input_file_path):
        # Load processed data
        try:
            df = pd.read_csv(input_file_path)
            # Fill NaNs in text columns that might affect embedding/storage
            for col in ['message_input', 'message_output', 'final_message_sent', 'sender_action', 'sender_id', 'job_id']:
                 if col in df.columns:
                      df[col] = df[col].fillna('N/A').astype(str)
            # Ensure numeric/date columns have appropriate handling for NaN if needed by metadata
            if 'detected_intent' in df.columns:
                df['detected_intent'] = pd.to_numeric(df['detected_intent'], errors='coerce').fillna(-1).astype(int)
            if 'edit_distance' in df.columns:
                df['edit_distance'] = pd.to_numeric(df['edit_distance'], errors='coerce').fillna(0.0).astype(float)
            if 'created_at' in df.columns:
                 df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')


            print(f"Loaded {len(df)} records for embedding.")

            # Initialize embedding model
            model = init_embedding_model() # Uses default model

            # Set up ChromaDB
            client, collections = setup_chroma_db(db_path=output_db_path)

            # Check if collections were actually created
            if not collections:
                 print("Error: Failed to create ChromaDB collections. Aborting storage.")
            else:
                 # Store data in ChromaDB
                 store_in_chroma(df, collections, model)

                 print("\nStandalone script: Vector database preparation complete!")

                 # Optional: Verify data was stored correctly
                 print("\nVerifying collection counts:")
                 for collection_name, collection in collections.items():
                     try:
                         count = collection.count()
                         print(f"- Collection '{collection_name}' contains {count} entries.")
                     except Exception as count_err:
                         print(f"- Error counting collection '{collection_name}': {count_err}")
        except FileNotFoundError:
             print(f"Error: Input file not found at {input_file_path}")
        except Exception as load_err:
             print(f"Error loading or processing data from {input_file_path}: {load_err}")

    else:
        print(f"Error: Input file not found at {input_file_path}")
        print("Please ensure the 'preprocess' step ran successfully and created './output/comparison_dataset.csv'")