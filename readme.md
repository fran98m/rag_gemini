# Sistema de Análisis RAG para Soporte al Cliente

Este sistema utiliza RAG (Generación Aumentada por Recuperación) con la API de Gemini 2.0 para analizar datos de soporte al cliente en español y mejorar la tasa de aceptación de sugerencias de IA.

## Características Principales

- **Soporte completo para español mexicano**: Optimizado para analizar contenido en español con consideraciones culturales y lingüísticas específicas para México
- **Análisis temporal avanzado**: Descubre patrones por día, día de la semana, hora y agente
- **Base de datos vectorial local**: Utiliza ChromaDB para almacenamiento vectorial gratuito sin necesidad de servicios externos
- **Integración con Gemini 2.0**: Utiliza el más reciente modelo gemini-2.0-flash para análisis de alta calidad
- **Visualizaciones interactivas**: Gráficos para tendencias temporales, clusters de rechazo y métricas de rendimiento

## Prerrequisitos

- Python 3.8+
- Clave de API de Gemini (nivel gratuito)
- Datos de soporte al cliente en formato CSV

## Instalación

1. **Clonar el repositorio o configurar tu directorio de proyecto**:

```bash
mkdir soporte-cliente-rag
cd soporte-cliente-rag
```

2. **Copiar todos los scripts de Python a tu directorio de proyecto**:
   - `data_preprocessing.py`
   - `embeddings_vectordb.py`
   - `gemini_analysis_updated.py`
   - `insight_generation_updated.py`
   - `time_analysis.py`
   - `main.py`

3. **Crear un entorno virtual e instalar dependencias**:

```bash
python -m venv rag_env
source rag_env/bin/activate  # En Windows: rag_env\Scripts\activate

# Instalar paquetes necesarios
pip install pandas numpy scikit-learn sentence-transformers 
pip install chromadb google-generativeai tqdm plotly
pip install psycopg2-binary  # Para la opción de PostgreSQL
```

4. **Configurar tu clave de API de Gemini**:

```bash
# En Linux/Mac
export GEMINI_API_KEY="tu_clave_api_aquí"

# En Windows (Command Prompt)
set GEMINI_API_KEY=tu_clave_api_aquí

# En Windows (PowerShell)
$env:GEMINI_API_KEY="tu_clave_api_aquí"
```

Alternativamente, puedes pasar la clave de API como argumento de línea de comandos al ejecutar el sistema.

## Uso del Sistema

El sistema está diseñado para ejecutarse como un pipeline con cinco pasos principales:

1. **Preprocesamiento de datos**: Limpia y prepara los datos
2. **Generación de embeddings**: Crea embeddings vectoriales y almacena en ChromaDB
3. **Análisis con API Gemini**: Analiza patrones de rechazo usando Gemini
4. **Generación de insights**: Agrupa rechazos y genera insights accionables
5. **Análisis temporal**: Analiza patrones por día, día de la semana y hora del día

### Ejecutar el Pipeline Completo

Para ejecutar todos los pasos con parámetros predeterminados:

```bash
python main.py --input_file ruta/a/tus/datos.csv --api_key tu_clave_api_aquí
```

### Ejecutar Pasos Individuales

Para ejecutar pasos específicos:

```bash
# Solo preprocesamiento
python main.py --input_file ruta/a/tus/datos.csv --steps preprocess

# Solo embeddings
python main.py --input_file ruta/a/tus/datos.csv --steps embed

# Solo análisis
python main.py --input_file ruta/a/tus/datos.csv --api_key tu_clave_api_aquí --steps analyze

# Solo generación de insights
python main.py --input_file ruta/a/tus/datos.csv --api_key tu_clave_api_aquí --steps insights

# Solo análisis temporal
python main.py --input_file ruta/a/tus/datos.csv --steps time

# Análisis temporal para un rango de fechas específico
python main.py --input_file ruta/a/tus/datos.csv --steps time --date_range 2023-01-01:2023-01-31
```

### Argumentos de Línea de Comandos

- `--input_file`: Ruta al archivo CSV de entrada (requerido)
- `--api_key`: Clave de API de Gemini (opcional si la variable de entorno está configurada)
- `--db_path`: Ruta para almacenamiento de ChromaDB (predeterminado: `./chroma_db`)
- `--output_dir`: Directorio para archivos de salida (predeterminado: `./output`)
- `--sample_size`: Número de muestras a analizar (predeterminado: 5000, usar -1 para todos)
- `--steps`: Qué pasos ejecutar (`all`, `preprocess`, `embed`, `analyze`, `insights`, `time`)
- `--date_range`: Filtro opcional de rango de fechas en formato YYYY-MM-DD:YYYY-MM-DD

### Ejemplos

```bash
# Ejecutar el pipeline completo
python main.py --input_file datos_soporte_cliente.csv --api_key tu_clave_aquí --output_dir ./resultados_analisis --sample_size 10000

# Ejecutar solo análisis temporal para el mes pasado
python main.py --input_file datos_soporte_cliente.csv --steps time --date_range 2023-03-01:2023-03-31 --output_dir ./analisis_marzo

# Ejecutar análisis en tiendas específicas (después de preprocesamiento con filtrado)
python main.py --input_file datos_tienda_A.csv --api_key tu_clave_aquí --output_dir ./resultados_tienda_A
```

## Formato de Datos de Entrada

Tu archivo CSV debe contener las siguientes columnas:

- `id`: Identificador único para cada mensaje
- `job_id`: Identificador para el trabajo de soporte al cliente
- `detected_intent`: La intención detectada por el sistema de IA (0-16)
- `message_input`: Mensaje del cliente
- `message_output`: Respuesta sugerida por la IA
- `sender_action`: Acción tomada por el agente humano (`accepted`, `ignored`, `edited`, `skipped`)
- `final_message_sent`: El mensaje real enviado al cliente
- `sender_id`: Identificador para el agente humano
- `created_at`: Marca de tiempo cuando se creó el mensaje
- `synced_at`: Marca de tiempo cuando se sincronizó el mensaje

## Archivos de Salida

El sistema genera varios archivos de salida:

- `dataset_stats.txt`: Estadísticas básicas sobre el dataset
- `analysis_dataset.csv`: Dataset procesado para análisis
- `comparison_dataset.csv`: Dataset enfocado en comparar respuestas sugeridas y finales
- `rejection_analysis.csv`: Resultados del análisis de API Gemini
- `cluster_visualization.html`: Visualización interactiva de clusters de rechazo
- `rejection_categories.html`: Distribución de categorías de rechazo
- `system_insights.json`: Insights generados por Gemini en formato JSON
- `improvement_report.md`: Informe final con recomendaciones accionables

### Salidas de Análisis Temporal

- `daily_stats.csv`: Estadísticas agregadas por día
- `weekday_stats.csv`: Estadísticas agregadas por día de la semana
- `hourly_stats.csv`: Estadísticas agregadas por hora del día
- `daypart_stats.csv`: Estadísticas agregadas por parte del día
- `agent_stats.csv`: Estadísticas agregadas por agente
- `weekly_stats.csv`: Estadísticas agregadas por semana del año
- `daily_acceptance_trend.html`: Visualización de tasas de aceptación diarias
- `weekday_acceptance_rate.html`: Visualización de tasas de aceptación por día de la semana
- `hourly_acceptance_rate.html`: Visualización de tasas de aceptación por hora
- `time_analysis_report.md`: Informe completo sobre patrones temporales

## Uso de ChromaDB

ChromaDB es una base de datos vectorial gratuita que se ejecuta localmente:
- Es un proyecto de código abierto sin costos de licencia
- Todos los datos se almacenan en un directorio local en tu máquina
- No requiere infraestructura en la nube o servicios externos
- Instalación simple a través de pip
- Bajos requisitos de recursos para el tamaño de dataset descrito

## Usando PostgreSQL en lugar de ChromaDB

Si prefieres usar PostgreSQL con pgvector en lugar de ChromaDB:

1. Instala PostgreSQL y la extensión pgvector
2. Modifica el archivo `embeddings_vectordb.py` para usar PostgreSQL
3. Actualiza los detalles de conexión en el script

## Trabajando con Recursos Limitados

Si tienes recursos computacionales limitados:

1. Usa un tamaño de muestra más pequeño: `--sample_size 1000`
2. Procesa en etapas: Ejecuta cada paso por separado
3. Usa modelos de embedding más simples

## Solución de Problemas

- **Límites de tasa de API**: El nivel gratuito de Gemini tiene límites de tasa. El sistema incluye lógica de reintento, pero puede que necesites ajustar tamaños de lote y añadir retrasos.
- **Problemas de memoria**: Reduce los tamaños de lote para la generación de embeddings si encuentras problemas de memoria.
- **Errores de ChromaDB**: Verifica que el directorio de persistencia exista y tenga los permisos adecuados.
- **Problemas con contenido en español**: Asegúrate de guardar archivos con codificación UTF-8 para manejar correctamente caracteres especiales.

## Extendiendo el Sistema

Este sistema RAG puede extenderse de varias maneras:

1. **Añadir soporte para más fuentes de datos**: Modifica el módulo de preprocesamiento para manejar diferentes formatos.
2. **Implementar análisis en tiempo real**: Añade un endpoint de API para analizar nuevos mensajes a medida que llegan.
3. **Crear un ciclo de retroalimentación**: Permite a los agentes humanos proporcionar retroalimentación explícita sobre los resultados del análisis.
4. **Añadir visualizaciones más avanzadas**: Integrar con Dash u otros frameworks de visualización.
5. **Entrenar modelos personalizados**: Usa los resultados para entrenar modelos personalizados para tu caso de uso específico.

## Próximos Pasos

Después de revisar el análisis:

1. Implementar las recomendaciones en el informe de mejora
2. Monitorear las tasas de aceptación después de los cambios
3. Ejecutar el análisis periódicamente para medir el progreso
4. Considerar usar los insights para reentrenar tu modelo de IA