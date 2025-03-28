import pandas as pd
import numpy as np
import json
import os
import re # Added for JSON extraction
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import datetime
import plotly.express as px
# import plotly.graph_objects as go # No longer used directly here
from sentence_transformers import SentenceTransformer

# Import the updated Gemini API components
# import base64 # No longer used
from google import genai
from google.genai import types

# --- Configuration Constants ---
DEFAULT_N_CLUSTERS = 5
# File names (relative to output_dir)
CLUSTER_VIZ_FILENAME = "cluster_visualization.html"
CATEGORY_VIZ_FILENAME = "rejection_categories.html"
INSIGHTS_JSON_FILENAME = "system_insights.json"
RAW_INSIGHTS_FILENAME = "raw_gemini_insights.txt"
REPORT_FILENAME = "improvement_report.md" # Matches main.py
# --- End Configuration Constants ---


# Load and prepare analysis data
def load_analysis_data(file_path: str) -> pd.DataFrame | None: # Added type hints
    """Load and prepare the rejection analysis data."""
    print(f"Loading analysis data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        # Clean up data if needed
        if 'rejection_category' in df.columns:
            df['rejection_category'] = df['rejection_category'].fillna('desconocido')
        # Ensure cluster column (if exists from previous runs) is numeric
        if 'cluster' in df.columns:
             df['cluster'] = pd.to_numeric(df['cluster'], errors='coerce') # Coerce errors to NaN
        return df
    except FileNotFoundError:
        print(f"Error: Analysis file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading analysis data from {file_path}: {e}")
        return None


# Cluster rejection reasons
def cluster_rejection_reasons(df: pd.DataFrame, n_clusters: int = DEFAULT_N_CLUSTERS) -> tuple[pd.DataFrame, dict]: # Added type hints
    """
    Cluster rejection reasons to identify patterns.

    Args:
        df: DataFrame with rejection analysis.
        n_clusters: Number of clusters to create.

    Returns:
        Tuple: (DataFrame with cluster assignments, Dictionary of top terms for each cluster)
    """
    print("Clustering rejection reasons...")

    # Check if 'rejection_reason' column exists and has data
    if 'rejection_reason' not in df.columns or df['rejection_reason'].isnull().all():
        print("WARN: 'rejection_reason' column not found or is empty. Cannot perform clustering.")
        df['cluster'] = -1 # Assign a default cluster ID
        return df, {} # Return empty cluster terms

    # Extract rejection reasons for clustering
    reasons = df['rejection_reason'].fillna('Desconocido').tolist()

    # Create TF-IDF vectors
    # Consider tuning TF-IDF parameters (e.g., max_df, ngram_range)
    vectorizer = TfidfVectorizer(
        max_features=1000, # Increased max_features for potentially better clustering
        stop_words=['de', 'la', 'el', 'en', 'y', 'a', 'que', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'es', 'más', 'este', 'le', 'lo', 'mi', 'o'], # Expanded stop words
        min_df=3, # Increased min_df
        ngram_range=(1, 2) # Include bigrams
    )

    try:
        # Transform text to vectors
        reason_vectors = vectorizer.fit_transform(reasons)

        # Perform clustering
        # Consider trying different clustering algorithms (e.g., DBSCAN, AgglomerativeClustering)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Explicitly set n_init
        df['cluster'] = kmeans.fit_predict(reason_vectors)

        # Get top terms for each cluster
        feature_names = vectorizer.get_feature_names_out()
        cluster_centers = kmeans.cluster_centers_

        cluster_terms = {}
        print("\nTérminos principales por cluster:")
        for i in range(n_clusters):
            # Get indices of top terms for this cluster
            top_indices = np.argsort(cluster_centers[i])[::-1][:10] # Get top 10, descending order
            top_terms = [feature_names[j] for j in top_indices]
            cluster_terms[i] = top_terms
            print(f"  Cluster {i}: {', '.join(top_terms)}")

    except Exception as e:
        print(f"Error during clustering: {e}")
        print("Skipping clustering visualization and assigning default cluster.")
        df['cluster'] = -1
        return df, {}

    return df, cluster_terms


# Generate cluster visualization
def visualize_clusters(df: pd.DataFrame, cluster_terms: dict, output_dir: str): # Added type hints
    """
    Create interactive visualizations for the clusters.

    Args:
        df: DataFrame with cluster assignments.
        cluster_terms: Dictionary of top terms for each cluster.
        output_dir: Directory to save HTML visualizations.
    """
    print("Generating cluster visualizations...")

    # Check if clustering was successful (cluster != -1)
    if 'cluster' not in df.columns or (df['cluster'] == -1).all():
         print("Skipping visualization as clustering was not performed or failed.")
         return df # Return unmodified DataFrame

    # Check if 'rejection_reason' exists for embeddings
    if 'rejection_reason' not in df.columns or df['rejection_reason'].isnull().all():
         print("Skipping visualization as 'rejection_reason' column is missing or empty.")
         return df

    # Define paths
    cluster_viz_path = os.path.join(output_dir, CLUSTER_VIZ_FILENAME)
    category_viz_path = os.path.join(output_dir, CATEGORY_VIZ_FILENAME)

    try:
        # Create a text embedding model (Consider making model name configurable)
        # Using a multilingual model is good for potential language variations
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        # Generate embeddings for the rejection reasons
        reasons = df['rejection_reason'].fillna('Desconocido').tolist()
        embeddings = model.encode(reasons, show_progress_bar=True) # Show progress

        # Reduce to 2D for visualization using PCA
        # Consider other dimensionality reduction techniques like t-SNE or UMAP for potentially better separation
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)

        # Add 2D coordinates to dataframe
        df['x_pca'] = embeddings_2d[:, 0]
        df['y_pca'] = embeddings_2d[:, 1]

        # Create cluster visualization
        # Ensure hover_data columns exist
        hover_cols = ['rejection_reason']
        if 'rejection_category' in df.columns:
            hover_cols.append('rejection_category')

        fig = px.scatter(
            df,
            x='x_pca',
            y='y_pca',
            color='cluster', # Use the cluster column
            color_continuous_scale=px.colors.qualitative.Plotly, # Use a qualitative color scale
            hover_data={col: True for col in hover_cols}, # Use dictionary for hover data
            title='Clusters de Razones de Rechazo (PCA)'
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7)) # Adjust marker size/opacity

        # Add cluster label annotations (optional, can clutter the plot)
        # Consider adding annotations only if n_clusters is small
        # for cluster, terms in cluster_terms.items():
        #     cluster_points = df[df['cluster'] == cluster]
        #     if not cluster_points.empty:
        #         center_x = cluster_points['x_pca'].mean()
        #         center_y = cluster_points['y_pca'].mean()
        #         fig.add_annotation(...) # Add annotation logic here if desired

        # Save interactive visualization
        fig.write_html(cluster_viz_path)
        print(f"Cluster visualization saved to {cluster_viz_path}")

        # Create rejection category distribution if column exists
        if 'rejection_category' in df.columns and not df['rejection_category'].isnull().all():
            fig2 = px.histogram(
                df,
                x='rejection_category',
                color='cluster', # Color by cluster
                barmode='group', # Or 'stack'
                title='Categorías de Rechazo por Cluster',
                category_orders={"cluster": sorted(df['cluster'].unique())} # Ensure consistent cluster order
            )
            fig2.update_xaxes(categoryorder='total descending') # Order categories by count
            fig2.write_html(category_viz_path)
            print(f"Category visualization saved to {category_viz_path}")

    except Exception as e:
        print(f"Error during cluster visualization: {e}")
        # Avoid modifying df further if visualization fails
        # Just print the error and continue

    return df


# Generate insights using Gemini
def generate_insights(client: genai.Client, df: pd.DataFrame, cluster_terms: dict, output_dir: str) -> dict | None: # Added type hints
    """
    Use Gemini to generate actionable insights based on the analysis.

    Args:
        client: The initialized Gemini API client (genai.Client).
        df: DataFrame with analysis results including cluster assignments.
        cluster_terms: Dictionary of top terms for each cluster.
        output_dir: Directory to save output files.

    Returns:
        Generated insights as a dictionary, or None on error.
    """
    print("Generating insights with Gemini...")
    model_name = "gemini-1.5-flash-latest" # Or pro, consider config

    # Define paths
    insights_json_path = os.path.join(output_dir, INSIGHTS_JSON_FILENAME)
    raw_insights_path = os.path.join(output_dir, RAW_INSIGHTS_FILENAME)


    # Check if clustering was performed
    if 'cluster' not in df.columns or df['cluster'].isnull().all() or (df['cluster'] == -1).all():
        print("WARN: Clustering information missing or invalid. Insights might be less specific.")
        # Optionally, adapt the prompt if clusters are missing

    # Prepare summary statistics
    cluster_counts = df['cluster'].value_counts().to_dict() if 'cluster' in df.columns else {}
    rejection_categories = df['rejection_category'].value_counts().to_dict() if 'rejection_category' in df.columns else {}

    # Get representative examples from each cluster
    examples_per_cluster = 3
    examples = {}
    if 'cluster' in df.columns:
        for cluster_id in df['cluster'].unique():
            if cluster_id == -1: continue # Skip invalid cluster
            cluster_df = df[df['cluster'] == cluster_id].copy()
            if not cluster_df.empty:
                sample_size = min(examples_per_cluster, len(cluster_df))
                sample = cluster_df.sample(sample_size, random_state=42)
                # Select relevant columns, ensure they exist
                example_cols = ['customer_message', 'suggested_response', 'final_response', 'rejection_reason']
                present_cols = [col for col in example_cols if col in sample.columns]
                examples[int(cluster_id)] = sample[present_cols].to_dict('records')


    # Create prompt for Gemini
    prompt = """
    Eres un analista experto revisando datos de un sistema de soporte al cliente para una empresa mexicana de comercio electrónico de alimentos. Se analizaron las razones por las cuales las sugerencias de respuesta de una IA fueron rechazadas o editadas por agentes humanos. Los rechazos se agruparon en clusters.

    Basándote en los datos resumidos a continuación (clusters, términos principales, ejemplos y distribución de categorías), genera insights accionables para mejorar drásticamente la tasa de aceptación de las sugerencias de la IA.

    Considera el contexto cultural mexicano y el lenguaje específico (español mexicano) para el comercio electrónico de alimentos.

    Información Disponible:
    """
    # Add cluster info if available
    if cluster_terms and cluster_counts:
         prompt += "\nClusters de Rechazo:\n"
         for cluster_id, terms in cluster_terms.items():
             count = cluster_counts.get(cluster_id, 0)
             if count == 0: continue # Skip empty clusters
             percentage = count / len(df) * 100 if len(df) > 0 else 0
             prompt += f"\n  Cluster {cluster_id} ({count} ejemplos, {percentage:.1f}% del total analizado):\n"
             prompt += f"  - Términos Clave: {', '.join(terms)}\n"

             # Add example rejections for this cluster
             if int(cluster_id) in examples:
                 prompt += "  - Ejemplos Representativos:\n"
                 for i, example in enumerate(examples[int(cluster_id)]):
                     prompt += f"    Ejemplo {i+1}:\n"
                     prompt += f"      Cliente: {example.get('customer_message', 'N/A')}\n"
                     prompt += f"      Sugerencia IA: {example.get('suggested_response', 'N/A')}\n"
                     prompt += f"      Respuesta Final Agente: {example.get('final_response', 'N/A')}\n"
                     prompt += f"      Razón (Análisis Previo): {example.get('rejection_reason', 'N/A')}\n"
    else:
         prompt += "\n(No se generó información detallada de clusters)\n"

    # Add category distribution if available
    if rejection_categories:
        prompt += "\nDistribución General de Categorías de Rechazo:\n"
        total_cat_count = sum(rejection_categories.values())
        for category, count in rejection_categories.items():
            percentage = count / total_cat_count * 100 if total_cat_count > 0 else 0
            prompt += f"- {category}: {count} ({percentage:.1f}%)\n"

    prompt += """

    Objetivo: Proporcionar un plan claro para mejorar la IA.

    Instrucciones de Respuesta:
    Genera una respuesta ÚNICAMENTE en formato JSON válido, siguiendo esta estructura exacta:
    {
        "resumen_ejecutivo": "Un breve resumen (2-3 frases) del problema principal y la oportunidad.",
        "hallazgos_clave": [
            "Hallazgo principal 1 sobre por qué fallan las sugerencias.",
            "Hallazgo principal 2...",
            "Hallazgo principal 3..."
        ],
        "recomendaciones_priorizadas": [
            {
                "recomendacion": "Recomendación accionable #1 (la más impactante).",
                "impacto_esperado": "Alto/Medio/Bajo",
                "esfuerzo_estimado": "Alto/Medio/Bajo",
                "clusters_relacionados": [0, 2] // Lista de IDs de clusters más afectados, si aplica
            },
            {
                "recomendacion": "Recomendación accionable #2.",
                "impacto_esperado": "...",
                "esfuerzo_estimado": "...",
                "clusters_relacionados": [...]
            }
            // Incluir de 3 a 5 recomendaciones clave
        ],
        "mejoras_especificas_por_cluster": {
            "0": [ // ID del cluster como string
                "Mejora específica 1 para el cluster 0.",
                "Mejora específica 2 para el cluster 0."
            ],
            "1": [
                "Mejora específica 1 para el cluster 1."
            ]
            // Incluir entradas para cada cluster relevante con >0 ejemplos
        },
        "sugerencias_sistema_intencion": [
            "Sugerencia 1 para mejorar la detección de intenciones, si aplica.",
            "Sugerencia 2..."
        ],
        "proximos_pasos_sugeridos": [
            "Paso 1 para implementar las mejoras.",
            "Paso 2..."
        ]
    }

    Asegúrate de que el JSON sea válido y completo según la estructura definida. No incluyas texto introductorio ni explicaciones fuera del JSON.
    """

    # Call Gemini API
    max_retries = 3
    for attempt in range(max_retries):
        try:
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
            generation_config = types.GenerationConfig(
                 response_mime_type="application/json", # Request JSON directly
                 # temperature=0.3 # Adjust temperature for insight generation
            )
            """
            safety_settings=[
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
            ]
            """
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                #config=generation_config,
                #safety_settings=safety_settings
            )

            # Since we requested JSON, response.text *should* be JSON
            # However, the API might still wrap it or add text if generation fails partially.
            insights_text = response.text
            insights = None

            try:
                # Attempt to parse directly
                 insights = json.loads(insights_text)
            except json.JSONDecodeError:
                 # If direct parsing fails, try extracting from markdown block
                 json_pattern = r'```json\s*([\s\S]*?)\s*```'
                 match = re.search(json_pattern, insights_text, re.DOTALL)
                 if match:
                     try:
                         insights = json.loads(match.group(1))
                     except json.JSONDecodeError as e:
                         print(f"Error parsing JSON even after extracting from block: {e}")
                         # Save raw response for debugging
                         with open(raw_insights_path, "w", encoding='utf-8') as f:
                             f.write(insights_text)
                         raise # Re-raise the error after saving raw output
                 else:
                     print("Error: Gemini response was not valid JSON and no JSON block found.")
                     # Save raw response for debugging
                     with open(raw_insights_path, "w", encoding='utf-8') as f:
                         f.write(insights_text)
                     raise json.JSONDecodeError("No valid JSON found in Gemini response.", insights_text, 0)


            # Save insights JSON
            with open(insights_json_path, "w", encoding='utf-8') as f:
                json.dump(insights, f, indent=2, ensure_ascii=False)

            print(f"Insights saved to {insights_json_path}")
            return insights # Success

        except Exception as e:
            print(f"Error generating insights on attempt {attempt + 1}: {type(e).__name__} - {e}")
            if attempt < max_retries - 1:
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print("Max retries reached for insight generation.")
                # Optionally save the last error or raw prompt/response
                return None # Indicate failure


# Generate final report
def generate_report(insights: dict, df: pd.DataFrame, output_dir: str): # Added type hints
    """
    Generate a final report in Markdown format.

    Args:
        insights: Generated insights dictionary from Gemini.
        df: Analysis DataFrame (used for counts, optional).
        output_dir: Directory to save the report.
    """
    report_path = os.path.join(output_dir, REPORT_FILENAME)
    print(f"Generating final report: {report_path}")

    # Safely get data from insights dictionary using .get()
    resumen = insights.get("resumen_ejecutivo", "N/A")
    hallazgos = insights.get("hallazgos_clave", [])
    recomendaciones = insights.get("recomendaciones_priorizadas", [])
    mejoras_cluster = insights.get("mejoras_especificas_por_cluster", {})
    mejoras_intencion = insights.get("sugerencias_sistema_intencion", [])
    proximos_pasos = insights.get("proximos_pasos_sugeridos", [])

    # Calculate basic overall stats if df is available
    total_analyzed = len(df) if df is not None else "N/A"

    # Generate markdown report string
    report = f"""# Informe de Mejora: Sistema de Sugerencias IA para Soporte

**Fecha:** {datetime.now().strftime('%Y-%m-%d')}

## Resumen Ejecutivo

{resumen}

*Este informe se basa en el análisis de {total_analyzed} interacciones donde las sugerencias de la IA fueron rechazadas o editadas.*

## Hallazgos Clave

"""
    if hallazgos:
        for i, finding in enumerate(hallazgos):
            report += f"{i+1}. {finding}\n"
    else:
        report += "*No se generaron hallazgos clave.*\n"

    report += "\n## Recomendaciones Priorizadas\n\n"
    if recomendaciones:
        report += "| Recomendación | Impacto Esperado | Esfuerzo Estimado | Clusters Relacionados |\n"
        report += "|---|---|---|---|\n"
        for rec in recomendaciones:
            clusters_str = ", ".join(map(str, rec.get("clusters_relacionados", [])))
            report += f"| {rec.get('recomendacion', 'N/A')} | {rec.get('impacto_esperado', 'N/A')} | {rec.get('esfuerzo_estimado', 'N/A')} | {clusters_str} |\n"
    else:
        report += "*No se generaron recomendaciones priorizadas.*\n"

    report += "\n## Mejoras Específicas por Cluster de Rechazo\n\n"
    if mejoras_cluster:
        for cluster_id, mejoras in mejoras_cluster.items():
            report += f"### Cluster {cluster_id}\n\n"
            if mejoras:
                for mejora in mejoras:
                    report += f"- {mejora}\n"
            else:
                report += "*No hay mejoras específicas para este cluster.*\n"
            report += "\n"
    else:
        report += "*No se generaron mejoras específicas por cluster.*\n"


    report += "## Sugerencias para el Sistema de Detección de Intenciones\n\n"
    if mejoras_intencion:
        for sugerencia in mejoras_intencion:
            report += f"- {sugerencia}\n"
    else:
        report += "*No se generaron sugerencias específicas para la detección de intenciones.*\n"


    report += "\n## Próximos Pasos Sugeridos\n\n"
    if proximos_pasos:
        for i, paso in enumerate(proximos_pasos):
            report += f"{i+1}. {paso}\n"
    else:
        report += "*No se generaron próximos pasos.*\n"


    # Save report
    try:
        with open(report_path, "w", encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved successfully to {report_path}")
    except Exception as e:
        print(f"Error saving report to {report_path}: {e}")

    # Note: This function doesn't *return* the report string currently,
    # but it could be modified to do so if needed elsewhere.


# Main function to run the insight generation pipeline
def run_insight_generation(client: genai.Client, analysis_df: pd.DataFrame, output_dir: str): # Added type hints
    """Run the complete insight generation pipeline."""
    print("\n--- Iniciando Generación de Insights ---")

    if analysis_df is None or analysis_df.empty:
         print("Error: El DataFrame de análisis está vacío o no se proporcionó. No se puede generar insights.")
         return None # Indicate failure

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Cluster rejection reasons
    df_clustered, cluster_terms = cluster_rejection_reasons(analysis_df)

    # 2. Generate visualizations (uses df_clustered)
    visualize_clusters(df_clustered, cluster_terms, output_dir)

    # 3. Generate insights using Gemini (uses df_clustered, cluster_terms, and the client)
    insights = generate_insights(client, df_clustered, cluster_terms, output_dir)

    # 4. Generate final report (uses insights and df_clustered)
    if insights:
        generate_report(insights, df_clustered, output_dir)
        print("--- Generación de Insights Completada ---")
        return insights # Return the generated insights
    else:
        print("Error: No se pudieron generar insights desde Gemini.")
        print("--- Generación de Insights Fallida ---")
        return None


# Example usage (for standalone testing, requires manual client setup)
if __name__ == "__main__":
    print("Running insight_generation.py as standalone script (for testing).")

    # --- Configuration for standalone run ---
    TEST_OUTPUT_DIR = "./output_test_insights"
    TEST_ANALYSIS_FILE = "./output/rejection_analysis.csv" # Assumes main.py ran 'analyze' step
    # --- End Configuration ---

    # IMPORTANT: For standalone testing, you MUST provide a valid API key.
    api_key_standalone = os.getenv("GEMINI_API_KEY") # Load from .env
    if not api_key_standalone:
        print("Error: GEMINI_API_KEY not found in environment for standalone run.")
        print("Please set it in your .env file or environment variables.")
    else:
        # Initialize client specifically for this test run
        test_client = genai.Client(api_key=api_key_standalone)
        print("Standalone: Gemini client initialized.")

        # Load analysis data
        test_analysis_df = load_analysis_data(TEST_ANALYSIS_FILE)

        if test_analysis_df is not None:
            # Run the insight generation pipeline
            run_insight_generation(test_client, test_analysis_df, TEST_OUTPUT_DIR)
        else:
            print(f"Standalone: Failed to load analysis data from {TEST_ANALYSIS_FILE}")