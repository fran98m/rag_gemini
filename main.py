import os
import argparse
import pandas as pd
import time
from datetime import datetime
from dotenv import load_dotenv # Added

# Updated imports for Gemini 2.0
from google import genai
# Remove unused import: from google.genai import types

# Import functions from our modules
from data_preprocessing import load_customer_support_data, explore_dataset, create_analysis_dataset
from embeddings_vectordb import init_embedding_model, setup_chroma_db, store_in_chroma
from time_analysis import run_time_analysis # Simplified import
import chromadb

# Import from analysis modules
from gemini_analysis import fetch_rejection_data, analyze_rejection_reasons # Removed setup_gemini_api, analyze_intent_accuracy (assuming not used currently), generate_system_insights (duplicate)
from insight_generation import load_analysis_data, run_insight_generation # Simplified import

# --- Configuration Constants ---
load_dotenv() # Load environment variables from .env file

DB_PATH_DEFAULT = './chroma_db'
OUTPUT_DIR_DEFAULT = './output'
SAMPLE_SIZE_DEFAULT = 5000
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Example constant for embedding model
# File names
DATASET_STATS_FILENAME = "dataset_stats.txt"
ANALYSIS_FILENAME = "analysis_dataset.csv"
COMPARISON_FILENAME = "comparison_dataset.csv"
RESULTS_FILENAME = "rejection_analysis.csv"
REPORT_FILENAME = "improvement_report.md"
TIME_REPORT_FILENAME = "time_analysis_report.md"
# --- End Configuration Constants ---

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Sistema de Análisis RAG para Soporte al Cliente')
    
    parser.add_argument('--input_file', type=str, required=True,
                        help='Ruta al archivo CSV de entrada')
    
    parser.add_argument('--api_key', type=str, default=None, # Made default None
                        help='Clave de API de Gemini (opcional, se prioriza sobre .env)')
    
    parser.add_argument('--db_path', type=str, default=DB_PATH_DEFAULT,
                        help='Ruta para almacenamiento de ChromaDB')
    
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR_DEFAULT,
                        help='Directorio para archivos de salida')
    
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE_DEFAULT,
                        help='Número de muestras a analizar (establecer a -1 para todas)')
    
    parser.add_argument('--steps', type=str, default='all',
                        choices=['all', 'preprocess', 'embed', 'analyze', 'insights', 'time'],
                        help='Qué pasos ejecutar')
    
    parser.add_argument('--date_range', type=str, default=None,
                        help='Rango de fechas para análisis en formato YYYY-MM-DD:YYYY-MM-DD')
    
    return parser.parse_args()

def get_gemini_client(args):
    """Initializes and returns the Gemini client using API key from args or .env"""
    api_key = args.api_key or os.getenv("GEMINI_API_KEY") # Use os.getenv
    if not api_key:
        raise ValueError("La clave de API de Gemini debe proporcionarse a través de --api_key o la variable de entorno GEMINI_API_KEY en el archivo .env")
    
    # Initialize Gemini client
    client = genai.Client(api_key=api_key)
    print("Cliente de API Gemini 2.0 inicializado.")
    return client


def run_rag_system(args):
    """Run the RAG system with the specified arguments"""
    print(f"='='='='='='='='='='='='='='='='='='='='='='='='='")
    print(f"Sistema de Análisis RAG para Soporte al Cliente")
    print(f"Iniciando en: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"='='='='='='='='='='='='='='='='='='='='='='='='='")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Gemini client (centralized)
    gemini_client = get_gemini_client(args) # Use helper function
        
    # Determine which steps to run
    run_preprocess = args.steps in ['all', 'preprocess']
    run_embed = args.steps in ['all', 'embed']
    run_analyze = args.steps in ['all', 'analyze']
    run_insights = args.steps in ['all', 'insights']
    run_time = args.steps in ['all', 'time']
    
    # Define paths using constants and args.output_dir
    analysis_path = os.path.join(args.output_dir, ANALYSIS_FILENAME)
    comparison_path = os.path.join(args.output_dir, COMPARISON_FILENAME)
    results_path = os.path.join(args.output_dir, RESULTS_FILENAME)
    stats_path = os.path.join(args.output_dir, DATASET_STATS_FILENAME)
    report_path = os.path.join(args.output_dir, REPORT_FILENAME)
    time_report_path = os.path.join(args.output_dir, TIME_REPORT_FILENAME)
    
    # Keep track of loaded dataframes
    df = None
    comparison_df = None
    analysis_results_df = None
    
    # Step 1: Data Preprocessing
    if run_preprocess:
        print("\n=== Paso 1: Preprocesamiento de Datos ===")
        start_time = time.time()
        
        # Load the data
        df = load_customer_support_data(args.input_file)
        
        # Sample if requested
        if args.sample_size > 0 and len(df) > args.sample_size:
            print(f"Muestreando {args.sample_size} registros de {len(df)} totales")
            df = df.sample(args.sample_size, random_state=42)
        
        # Explore the dataset
        stats, samples = explore_dataset(df)
        
        # Save dataset statistics
        with open(stats_path, "w", encoding='utf-8') as f: # Use constant
            f.write("Estadísticas del Dataset:\n")
            for key, value in stats.items():
                if not isinstance(value, dict):
                    f.write(f"- {key}: {value}\n")
                else:
                    f.write(f"- {key}:\n")
                    for k, v in value.items():
                        f.write(f"  - {k}: {v}\n")
        
        # Create analysis datasets
        analysis_df_step1, comparison_df = create_analysis_dataset(df) # Use different variable name to avoid confusion if df is reused
        
        # Save processed datasets
        analysis_df_step1.to_csv(analysis_path, index=False)
        comparison_df.to_csv(comparison_path, index=False)
        
        elapsed = time.time() - start_time
        print(f"Preprocesamiento completado en {elapsed:.2f} segundos")
        print(f"Dataset de análisis: {len(analysis_df_step1)} registros")
        print(f"Dataset de comparación: {len(comparison_df)} registros")
        print(f"Archivos guardados en {args.output_dir}")
    
    # Step 2: Embeddings and Vector DB
    if run_embed:
        print("\n=== Paso 2: Embeddings y Base de Datos Vectorial ===")
        start_time = time.time()
        
        # Load the processed data if not already loaded from Step 1
        if comparison_df is None:
            if os.path.exists(comparison_path):
                print(f"Cargando datos procesados desde {comparison_path}")
                comparison_df = pd.read_csv(comparison_path)
            else:
                 print(f"Error: Archivo de comparación no encontrado en {comparison_path}")
                 print("El paso de embedding requiere datos procesados del Paso 1.")
                 return # Or handle error appropriately

        if comparison_df is not None: # Proceed only if data is loaded
            # Initialize embedding model (consider passing model name constant)
            model = init_embedding_model() # Maybe: init_embedding_model(EMBEDDING_MODEL_NAME)
            
            # Set up ChromaDB
            client_db, collections = setup_chroma_db(args.db_path) # Pass EMBEDDING_MODEL_NAME if needed by setup
            
            # Store data in ChromaDB
            store_in_chroma(comparison_df, collections, model)
            
            elapsed = time.time() - start_time
            print(f"Preparación de base de datos vectorial completada en {elapsed:.2f} segundos")
            
            # Verify data was stored correctly
            for collection_name, collection in collections.items():
                count = collection.count()
                print(f"Colección '{collection_name}' contiene {count} entradas")
        else:
             print("Omitiendo Paso 2 debido a datos de comparación faltantes.")
    
    # Step 3: Gemini API Analysis
    if run_analyze:
        print("\n=== Paso 3: Análisis con API Gemini 2.0 ===")
        start_time = time.time()
        
        # Connect to existing ChromaDB
        chroma_client = None
        try:
            chroma_client = chromadb.PersistentClient(path=args.db_path)
            print(f"Conectado a ChromaDB en {args.db_path}")
        except Exception as e:
            print(f"Error al conectar a ChromaDB: {str(e)}")
            print("El paso de análisis requiere la base de datos vectorial del Paso 2.")
            return # Or handle error appropriately
        
        # Fetch rejection data
        print("Obteniendo datos de rechazo desde ChromaDB...")
        rejection_data = fetch_rejection_data(chroma_client, limit=200)  # Adjust limit as needed
        print(f"Recuperados {len(rejection_data)} ejemplos de rechazo para análisis.")
        
        if rejection_data:
            # Analyze rejection reasons with Gemini 2.0
            print("Analizando razones de rechazo con API Gemini 2.0...")
            # Pass the initialized gemini_client
            analysis_results = analyze_rejection_reasons(gemini_client, rejection_data, batch_size=5)
            
            # Save analysis results
            analysis_results_df = pd.DataFrame(analysis_results)
            analysis_results_df.to_csv(results_path, index=False, encoding='utf-8')
            
            elapsed = time.time() - start_time
            print(f"Análisis completado en {elapsed:.2f} segundos.")
            print(f"Resultados guardados en {results_path}")
        else:
            print("No se encontraron datos de rechazo en ChromaDB. Omitiendo análisis.")
            elapsed = time.time() - start_time
            print(f"Paso de análisis omitido en {elapsed:.2f} segundos.")


    # Step 4: Generate Insights
    if run_insights:
        print("\n=== Paso 4: Generación de Insights ===")
        start_time = time.time()
        
        # Load analysis results if not already loaded from Step 3
        if analysis_results_df is None:
             if os.path.exists(results_path):
                 print(f"Cargando resultados de análisis desde {results_path}")
                 analysis_results_df = load_analysis_data(results_path) # Use function from insight_generation
             else:
                 print(f"Error: Resultados de análisis no encontrados en {results_path}")
                 print("El paso de insights requiere resultados de análisis del Paso 3.")
                 return # Or handle error appropriately
        
        if analysis_results_df is not None: # Proceed only if data is loaded
            # Run insight generation, passing the initialized gemini_client
            run_insight_generation(gemini_client, analysis_results_df, args.output_dir)
            
            elapsed = time.time() - start_time
            print(f"Generación de insights completada en {elapsed:.2f} segundos.")
            print(f"Informe guardado en {report_path}")
        else:
            print("Omitiendo Paso 4 debido a resultados de análisis faltantes.")

    
    # Step 5: Time-based Analysis
    if run_time:
        print("\n=== Paso 5: Análisis Basado en Tiempo ===")
        start_time = time.time()
        
        # Load main data if not already loaded from Step 1
        if df is None:
            if os.path.exists(analysis_path): # Use analysis_path which is created in step 1
                print(f"Cargando datos procesados desde {analysis_path}")
                df = pd.read_csv(analysis_path) # Load the analysis dataset from step 1 output
            # Optionally load original input if analysis_path doesn't exist but preprocessing was skipped
            elif not run_preprocess and os.path.exists(args.input_file):
                 print(f"Cargando datos originales desde {args.input_file} (el preprocesamiento fue omitido)")
                 df = load_customer_support_data(args.input_file) # Reload and preprocess minimally if needed
                 # Minimal preprocessing needed for time analysis might be just date conversion
                 if 'created_at' in df.columns and df['created_at'].dtype != 'datetime64[ns]':
                    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
                 if 'sender_action' in df.columns: # Need action flags for time analysis report
                     df['suggestion_accepted'] = df['sender_action'] == 'ACCEPTED'
                     df['suggestion_edited'] = df['sender_action'] == 'EDITED'
                     df['suggestion_ignored'] = df['sender_action'] == 'IGNORED'
                     df['suggestion_skipped'] = df['sender_action'] == 'SKIPPED'

            else:
                print(f"Error: Dataset no encontrado en {analysis_path} o {args.input_file}")
                print("El paso de análisis temporal requiere datos del Paso 1 o el archivo de entrada.")
                return # Or handle error appropriately
        
        if df is not None: # Proceed only if data is loaded
            df_for_time = df.copy() # Use a copy to avoid modifying the original df used by other steps

            # Apply date range filter if specified
            if args.date_range:
                try:
                    start_date, end_date = args.date_range.split(':')
                    # Ensure 'created_at' is datetime
                    if 'created_at' in df_for_time.columns and df_for_time['created_at'].dtype != 'datetime64[ns]':
                        df_for_time['created_at'] = pd.to_datetime(df_for_time['created_at'], errors='coerce')
                    
                    df_filtered = df_for_time[
                        (df_for_time['created_at'] >= start_date) &
                        (df_for_time['created_at'] <= end_date)
                    ].copy() # Use copy to avoid SettingWithCopyWarning

                    print(f"Datos filtrados al rango de fechas {start_date} a {end_date}")
                    print(f"Dataset filtrado contiene {len(df_filtered)} registros")
                    
                    if len(df_filtered) < 100:
                        print("¡Advertencia: Muy pocos registros en el rango de fechas especificado!")
                    
                    df_for_time = df_filtered # Use the filtered data
                except Exception as e:
                    print(f"Error al aplicar filtro de fechas: {str(e)}")
                    print("Usando el dataset completo en su lugar.")
            
            # Ensure required columns exist before running analysis
            required_cols = ['created_at', 'id', 'sender_action', 'sender_id', 'detected_intent']
            # Add action flags if missing (might happen if loaded directly from input)
            if 'suggestion_accepted' not in df_for_time.columns and 'sender_action' in df_for_time.columns:
                 df_for_time['suggestion_accepted'] = df_for_time['sender_action'] == 'ACCEPTED'
                 df_for_time['suggestion_edited'] = df_for_time['sender_action'] == 'EDITED'
                 df_for_time['suggestion_ignored'] = df_for_time['sender_action'] == 'IGNORED'
                 df_for_time['suggestion_skipped'] = df_for_time['sender_action'] == 'SKIPPED'

            missing_cols = [col for col in required_cols if col not in df_for_time.columns]
            action_flags = ['suggestion_accepted', 'suggestion_edited', 'suggestion_ignored', 'suggestion_skipped']
            missing_action_flags = [flag for flag in action_flags if flag not in df_for_time.columns]


            if not missing_cols and not missing_action_flags:
                # Run time-based analysis
                time_results = run_time_analysis(df_for_time, args.output_dir)
                
                elapsed = time.time() - start_time
                print(f"Análisis temporal completado en {elapsed:.2f} segundos.")
                print(f"Informe de análisis temporal guardado en {time_report_path}")
            else:
                 print(f"Error: Faltan columnas requeridas para análisis temporal: {missing_cols + missing_action_flags}")
                 print("Asegúrate de que el Paso 1 (Preprocesamiento) se haya ejecutado o que el archivo de entrada las contenga.")

        else:
            print("Omitiendo Paso 5 debido a datos faltantes.")

    
    print("\n=== Ejecución del Sistema de Análisis RAG Completada ===")
    print(f"Todas las salidas guardadas en: {args.output_dir}")

if __name__ == "__main__":
    args = parse_args()
    run_rag_system(args)