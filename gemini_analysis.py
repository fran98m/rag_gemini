import pandas as pd
# import numpy as np # No longer used directly here
import os
import json
import time
import random
import re
import chromadb
from tqdm import tqdm
# Updated imports for Gemini 2.0 API
from google import genai # Import genai directly
from google.genai import types



# Function to analyze a batch of rejection pairs
def analyze_rejection_reasons(client: genai.Client, rejection_pairs: list, batch_size: int = 5) -> list: # Added type hints
    """
    Use Gemini to analyze why suggested responses were rejected.

    Args:
        client: The initialized Gemini API client (genai.Client).
        rejection_pairs: List of dicts containing rejection data.
        batch_size: Number of examples to analyze in one batch.

    Returns:
        List of analysis results.
    """
    results = []
    model_name = "gemini-1.5-flash-latest" # Or use gemini-1.5-pro-latest, etc. Consider making this configurable.

    # Process in batches to manage rate limits
    for i in tqdm(range(0, len(rejection_pairs), batch_size), desc="Analyzing Rejections"):
        batch = rejection_pairs[i:i+batch_size]

        # Create detailed prompt for analysis
        prompt = """
        You are analyzing Spanish language customer support interactions for a Mexican client where AI-suggested responses were rejected or modified by human agents.

        For EACH example below, explain in Spanish:
        1. Why the AI suggestion might have been rejected.
        2. Key differences between the suggested and final responses.
        3. What could be improved about the AI suggestion.

        Pay special attention to:
        - Mexican Spanish idioms and expressions.
        - Cultural context and politeness norms in Mexican Spanish.
        - Tonality and formality appropriate for grocery e-commerce in Mexico.

        Provide your analysis as a JSON list, where each object in the list corresponds to one EXAMPLE provided below and has these fields:
        - id: The original ID corresponding to the example (use the index 'j' from the loop: 0, 1, 2...).
        - rejection_reason: The primary reason the suggestion was rejected (in Spanish).
        - improvement_areas: Specific areas where the suggestion could be improved (in Spanish).
        - intent_accuracy: Whether the detected intent appears to be correct (true/false).
        - rejection_category: Categorize the rejection (tono, información_faltante, inexacto, formato, error_de_intención, otro).

        JSON List structure:
        [
          { "id": 0, "rejection_reason": "...", "improvement_areas": "...", "intent_accuracy": true, "rejection_category": "..." },
          { "id": 1, "rejection_reason": "...", "improvement_areas": "...", "intent_accuracy": false, "rejection_category": "..." },
          ...
        ]

        BEGIN EXAMPLES:
        """

        for j, item in enumerate(batch):
            prompt += f"""
            EXAMPLE {j}:
            Customer message: {item.get('customer_message', 'N/A')}
            Detected intent: {item.get('detected_intent', 'N/A')}
            AI suggestion: {item.get('suggested_response', 'N/A')}
            Final response: {item.get('final_response', 'N/A')}
            Action: {item.get('action', 'N/A')}

            """
        prompt += "END EXAMPLES\nRespond ONLY with the JSON list."


        # Handle potential API errors and rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Create content structure for the request
                # Note: Gemini API expects specific content structures. Adjust if needed.
                contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]

                # Configure response as plain text (to extract JSON later)
                # Consider trying "application/json" if the model reliably returns JSON
                generation_config = types.GenerationConfig(
                    # response_mime_type="application/json", # Alternative
                    response_mime_type="text/plain",
                    # temperature=0.2 # Example: Adjust temperature if needed
                )

                # Safety settings (optional, adjust as needed)
                """
                safety_settings = [
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    ),
                     types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    ),
                    # Add other categories as needed
                ]
                """

                # Call the Gemini API using the provided client
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    #config=generation_config,
                    #safety_settings=safety_settings
                )

                analysis_text = response.text
                analysis_data = None # Initialize analysis_data

                # Extract JSON from response more robustly
                try:
                    # Try finding JSON block first
                    json_pattern = r'```json\s*([\s\S]*?)\s*```'
                    match = re.search(json_pattern, analysis_text, re.DOTALL)
                    if match:
                         analysis_data = json.loads(match.group(1))
                    else:
                         # If no block, try parsing the whole text directly
                         analysis_data = json.loads(analysis_text)

                except json.JSONDecodeError as json_err:
                     print(f"WARN: Failed to decode JSON from response batch starting at index {i}. Error: {json_err}")
                     print(f"      Response Text (first 200 chars): {analysis_text[:200]}...")
                     # Create error entries for this batch
                     analysis_data = [
                         {"id": j, "rejection_reason": "Error al extraer análisis JSON", "improvement_areas": [], "intent_accuracy": None, "rejection_category": "error_parsing"}
                         for j in range(len(batch))
                     ]


                # Ensure analysis_data is a list
                if not isinstance(analysis_data, list):
                    print(f"WARN: Expected JSON list, but received type {type(analysis_data)} for batch starting at index {i}. Creating error entries.")
                    analysis_data = [
                         {"id": j, "rejection_reason": "Error: Formato de respuesta inesperado (no es lista JSON)", "improvement_areas": [], "intent_accuracy": None, "rejection_category": "error_format"}
                         for j in range(len(batch))
                     ]

                # Add analysis to results, matching by original index 'j'
                analysis_map = {item.get("id", -1): item for item in analysis_data} # Map results by ID
                for j, original_item in enumerate(batch):
                    item_result = original_item.copy()
                    analysis_for_item = analysis_map.get(j) # Find result matching original index j

                    if analysis_for_item:
                        item_result.update(analysis_for_item)
                    else:
                        # If no matching ID found in response JSON list
                         print(f"WARN: No analysis found for item index {j} in batch starting at {i}.")
                         item_result.update({
                            "rejection_reason": "Error: Análisis faltante en respuesta JSON",
                            "improvement_areas": [],
                            "intent_accuracy": None, # Use None for unknown boolean
                            "rejection_category": "error_missing"
                         })
                    results.append(item_result)

                # Success, break retry loop
                break

            except Exception as e:
                # Catch potential API errors (rate limits, server errors, etc.)
                # Check specific error types if available from the google.api_core.exceptions module
                print(f"ERROR during Gemini API call on attempt {attempt+1} for batch starting at index {i}: {type(e).__name__} - {str(e)}")
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    sleep_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    print(f"Max retries reached for batch starting at index {i}. Skipping batch.")
                    # Add error analysis for skipped items in the batch
                    for original_item in batch:
                        item_result = original_item.copy()
                        item_result.update({
                            "rejection_reason": f"Error en llamada a API después de {max_retries} intentos",
                            "improvement_areas": [],
                            "intent_accuracy": None,
                            "rejection_category": "error_api"
                        })
                        results.append(item_result)
            # Optional short delay between batches to help with rate limiting
            # time.sleep(0.5)

    return results

# Removed analyze_intent_accuracy function (assuming not currently used or will be refactored separately)

# Removed generate_system_insights function (duplicate of generate_insights in insight_generation.py)


# Function to fetch rejection data from ChromaDB
def fetch_rejection_data(chroma_client: chromadb.Client, limit: int = 100) -> list: # Added type hints
    """
    Fetch a sample of rejected pairs from ChromaDB for analysis.

    Args:
        chroma_client: Initialized ChromaDB persistent client.
        limit: Maximum number of examples to retrieve.

    Returns:
        List of rejection pairs (dictionaries).
    """
    try:
        # Ensure collection name matches the one used in embeddings_vectordb.py
        collection = chroma_client.get_collection("rejected_pairs")
    except Exception as e:
        print(f"Error getting ChromaDB collection 'rejected_pairs': {e}")
        print("Ensure the 'embed' step ran successfully and the collection exists.")
        return [] # Return empty list on error

    # Get a sample of rejection data - Use get() instead of query() for sampling without a query vector
    try:
        result = collection.get(
            limit=limit,
            include=["documents", "metadatas"] # Specify what data to include
        )
    except Exception as e:
         print(f"Error fetching data from ChromaDB collection 'rejected_pairs': {e}")
         return []

    rejection_data = []
    # Check if 'documents' and 'metadatas' are not None and have the same length
    if result and result.get('documents') and result.get('metadatas') and len(result['documents']) == len(result['metadatas']):
        doc_list = result['documents']
        meta_list = result['metadatas']

        for i in range(len(doc_list)):
            doc = doc_list[i]
            meta = meta_list[i]

            # Check if doc and meta are not None before processing
            if doc is None or meta is None:
                 print(f"WARN: Skipping record {i} due to None document or metadata.")
                 continue

            # Parse the combined document into its components robustly
            customer_message = 'N/A'
            suggested_response = 'N/A'
            final_response = 'N/A'
            try:
                lines = doc.split('\n')
                for line in lines:
                    if line.startswith('CUSTOMER: '):
                        customer_message = line.replace('CUSTOMER: ', '', 1)
                    elif line.startswith('SUGGESTION: '):
                        suggested_response = line.replace('SUGGESTION: ', '', 1)
                    elif line.startswith('FINAL: '):
                        final_response = line.replace('FINAL: ', '', 1)
            except Exception as parse_err:
                 print(f"WARN: Could not parse document string for record {i}: {parse_err}. Document: '{doc[:100]}...'")


            # Create structured item using .get() for safer access
            item = {
                # Include original ID if available in Chroma metadata, otherwise use index or generate one
                'id': meta.get('id', f"chroma_{i}"), # Example: assuming 'id' might be stored
                'customer_message': customer_message,
                'suggested_response': suggested_response,
                'final_response': final_response,
                'detected_intent': meta.get('detected_intent'),
                'action': meta.get('sender_action'),
                'edit_distance': meta.get('edit_distance', 0.0) # Default to float
            }

            rejection_data.append(item)
    else:
        print("WARN: ChromaDB result format unexpected or empty.")
        if result:
            print(f"      Keys found: {result.keys()}")
            print(f"      Documents length: {len(result.get('documents', [])) if result.get('documents') else 'None'}")
            print(f"      Metadatas length: {len(result.get('metadatas', [])) if result.get('metadatas') else 'None'}")


    return rejection_data

# Removed the if __name__ == "__main__": block as setup_gemini_api was removed.
# Standalone execution of this script would need adjustment to receive a client object.