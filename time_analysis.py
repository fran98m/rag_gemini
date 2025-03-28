import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import calendar
import os

def prepare_time_data(df):
    """
    Prepare dataframe for time-based analysis
    """
    # Ensure datetime columns are properly formatted
    for col in ['created_at', 'synced_at']:
        if col in df.columns and df[col].dtype != 'datetime64[ns]':
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Create time-based features
    if 'created_at' in df.columns:
        df['date'] = df['created_at'].dt.date
        df['day_of_week'] = df['created_at'].dt.day_name()
        df['hour_of_day'] = df['created_at'].dt.hour
        df['week_of_year'] = df['created_at'].dt.isocalendar().week
        df['month'] = df['created_at'].dt.month
        df['month_name'] = df['created_at'].dt.month_name()
        
        # Create day part (morning, afternoon, evening, night)
        df['day_part'] = pd.cut(
            df['hour_of_day'],
            bins=[0, 6, 12, 18, 24],
            labels=['Madrugada', 'Mañana', 'Tarde', 'Noche'],
            include_lowest=True
        )
    
    return df

def analyze_by_day(df, output_dir="./output"):
    """
    Analyze acceptance rate and patterns by day
    """
    print("Analyzing patterns by day...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by date
    daily_stats = df.groupby('date').agg(
        total_messages=('id', 'count'),
        accepted=('suggestion_accepted', 'sum'),
        edited=('suggestion_edited', 'sum'),
        ignored=('suggestion_ignored', 'sum'),
        skipped=('suggestion_skipped', 'sum')
    ).reset_index()
    
    # Calculate acceptance rates
    daily_stats['acceptance_rate'] = (daily_stats['accepted'] / daily_stats['total_messages'] * 100).round(2)
    daily_stats['rejection_rate'] = 100 - daily_stats['acceptance_rate']
    
    # Create daily trend visualization
    fig = px.line(
        daily_stats, 
        x='date', 
        y='acceptance_rate',
        title='Tasa de Aceptación Diaria de Sugerencias de IA',
        labels={'date': 'Fecha', 'acceptance_rate': 'Tasa de Aceptación (%)'}
    )
    
    # Add moving average trendline (7-day)
    daily_stats['acceptance_ma7'] = daily_stats['acceptance_rate'].rolling(7, min_periods=1).mean()
    fig.add_scatter(
        x=daily_stats['date'], 
        y=daily_stats['acceptance_ma7'],
        mode='lines',
        name='Promedio Móvil (7 días)',
        line=dict(color='red', width=2, dash='dash')
    )
    
    # Add target line
    fig.add_shape(
        type="line",
        x0=daily_stats['date'].min(),
        y0=80,
        x1=daily_stats['date'].max(),
        y1=80,
        line=dict(color="green", width=2, dash="dot"),
    )
    fig.add_annotation(
        x=daily_stats['date'].max(),
        y=80,
        text="Meta (80%)",
        showarrow=False,
        yshift=10
    )
    
    # Save visualization
    fig.write_html(os.path.join(output_dir, "daily_acceptance_trend.html"))
    
    # Create daily volumes visualization
    fig2 = px.bar(
        daily_stats,
        x='date',
        y=['accepted', 'edited', 'ignored', 'skipped'],
        title='Volumen Diario por Tipo de Acción',
        labels={'date': 'Fecha', 'value': 'Número de Mensajes', 'variable': 'Acción'},
        color_discrete_map={
            'accepted': 'green',
            'edited': 'orange',
            'ignored': 'red',
            'skipped': 'gray'
        }
    )
    fig2.write_html(os.path.join(output_dir, "daily_action_volumes.html"))
    
    # Save daily stats CSV
    daily_stats.to_csv(os.path.join(output_dir, "daily_stats.csv"), index=False)
    
    return daily_stats

def analyze_by_day_of_week(df, output_dir="./output"):
    """
    Analyze patterns by day of week
    """
    print("Analyzing patterns by day of week...")
    
    # Group by day of week
    # Ensure proper ordering of days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_order_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    
    # Map English day names to Spanish
    day_map = dict(zip(day_order, day_order_es))
    df['day_of_week_es'] = df['day_of_week'].map(day_map)
    
    weekday_stats = df.groupby('day_of_week').agg(
        total_messages=('id', 'count'),
        accepted=('suggestion_accepted', 'sum'),
        edited=('suggestion_edited', 'sum'),
        ignored=('suggestion_ignored', 'sum'),
        skipped=('suggestion_skipped', 'sum')
    ).reset_index()
    
    # Add Spanish day names
    weekday_stats['day_of_week_es'] = weekday_stats['day_of_week'].map(day_map)
    
    # Ensure proper ordering
    weekday_stats['day_order'] = weekday_stats['day_of_week'].apply(lambda x: day_order.index(x))
    weekday_stats = weekday_stats.sort_values('day_order')
    
    # Calculate rates
    weekday_stats['acceptance_rate'] = (weekday_stats['accepted'] / weekday_stats['total_messages'] * 100).round(2)
    
    # Create day of week visualization
    fig = px.bar(
        weekday_stats,
        x='day_of_week_es',
        y='acceptance_rate',
        title='Tasa de Aceptación por Día de la Semana',
        labels={'day_of_week_es': 'Día de la Semana', 'acceptance_rate': 'Tasa de Aceptación (%)'},
        category_orders={"day_of_week_es": day_order_es}
    )
    
    # Add target line
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=80,
        x1=6.5,
        y1=80,
        line=dict(color="green", width=2, dash="dot"),
    )
    
    # Save visualization
    fig.write_html(os.path.join(output_dir, "weekday_acceptance_rate.html"))
    
    # Action breakdown by day of week
    fig2 = px.bar(
        weekday_stats,
        x='day_of_week_es',
        y=['accepted', 'edited', 'ignored', 'skipped'],
        title='Distribución de Acciones por Día de la Semana',
        labels={'day_of_week_es': 'Día de la Semana', 'value': 'Cantidad', 'variable': 'Acción'},
        category_orders={"day_of_week_es": day_order_es},
        color_discrete_map={
            'accepted': 'green',
            'edited': 'orange',
            'ignored': 'red',
            'skipped': 'gray'
        }
    )
    fig2.write_html(os.path.join(output_dir, "weekday_action_breakdown.html"))
    
    # Save weekday stats CSV
    weekday_stats.to_csv(os.path.join(output_dir, "weekday_stats.csv"), index=False)
    
    return weekday_stats

def analyze_by_hour(df, output_dir="./output"):
    """
    Analyze patterns by hour of day
    """
    print("Analyzing patterns by hour of day...")
    
    # Group by hour
    hourly_stats = df.groupby('hour_of_day').agg(
        total_messages=('id', 'count'),
        accepted=('suggestion_accepted', 'sum'),
        edited=('suggestion_edited', 'sum'),
        ignored=('suggestion_ignored', 'sum'),
        skipped=('suggestion_skipped', 'sum')
    ).reset_index()
    
    # Calculate rates
    hourly_stats['acceptance_rate'] = (hourly_stats['accepted'] / hourly_stats['total_messages'] * 100).round(2)
    
    # Create hour of day visualization
    fig = px.line(
        hourly_stats,
        x='hour_of_day',
        y='acceptance_rate',
        title='Tasa de Aceptación por Hora del Día',
        labels={'hour_of_day': 'Hora del Día (0-23)', 'acceptance_rate': 'Tasa de Aceptación (%)'},
        markers=True
    )
    
    # Add target line
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=80,
        x1=23.5,
        y1=80,
        line=dict(color="green", width=2, dash="dot"),
    )
    
    # Improve x-axis formatting
    fig.update_xaxes(tickmode='linear', tick0=0, dtick=2)
    
    # Save visualization
    fig.write_html(os.path.join(output_dir, "hourly_acceptance_rate.html"))
    
    # Message volume by hour
    fig2 = px.bar(
        hourly_stats,
        x='hour_of_day',
        y='total_messages',
        title='Volumen de Mensajes por Hora del Día',
        labels={'hour_of_day': 'Hora del Día (0-23)', 'total_messages': 'Cantidad de Mensajes'}
    )
    fig2.update_xaxes(tickmode='linear', tick0=0, dtick=2)
    fig2.write_html(os.path.join(output_dir, "hourly_message_volume.html"))
    
    # Save hourly stats CSV
    hourly_stats.to_csv(os.path.join(output_dir, "hourly_stats.csv"), index=False)
    
    # Day part analysis
    daypart_stats = df.groupby('day_part').agg(
        total_messages=('id', 'count'),
        accepted=('suggestion_accepted', 'sum'),
        edited=('suggestion_edited', 'sum'),
        ignored=('suggestion_ignored', 'sum'),
        skipped=('suggestion_skipped', 'sum')
    ).reset_index()
    
    # Calculate rates
    daypart_stats['acceptance_rate'] = (daypart_stats['accepted'] / daypart_stats['total_messages'] * 100).round(2)
    
    # Create day part visualization
    fig3 = px.bar(
        daypart_stats,
        x='day_part',
        y='acceptance_rate',
        title='Tasa de Aceptación por Parte del Día',
        labels={'day_part': 'Parte del Día', 'acceptance_rate': 'Tasa de Aceptación (%)'},
        category_orders={"day_part": ['Madrugada', 'Mañana', 'Tarde', 'Noche']}
    )
    
    # Add target line
    fig3.add_shape(
        type="line",
        x0=-0.5,
        y0=80,
        x1=3.5,
        y1=80,
        line=dict(color="green", width=2, dash="dot"),
    )
    
    # Save visualization
    fig3.write_html(os.path.join(output_dir, "daypart_acceptance_rate.html"))
    
    # Save day part stats CSV
    daypart_stats.to_csv(os.path.join(output_dir, "daypart_stats.csv"), index=False)
    
    return hourly_stats, daypart_stats

def analyze_by_agent(df, output_dir="./output"):
    """
    Analyze patterns by agent
    """
    print("Analyzing patterns by agent...")
    
    # Group by agent
    agent_stats = df.groupby('sender_id').agg(
        total_messages=('id', 'count'),
        accepted=('suggestion_accepted', 'sum'),
        edited=('suggestion_edited', 'sum'),
        ignored=('suggestion_ignored', 'sum'),
        skipped=('suggestion_skipped', 'sum')
    ).reset_index()
    
    # Calculate rates
    agent_stats['acceptance_rate'] = (agent_stats['accepted'] / agent_stats['total_messages'] * 100).round(2)
    
    # Sort by volume
    agent_stats = agent_stats.sort_values('total_messages', ascending=False)
    
    # Create top agents visualization (limited to top 20 by volume)
    top_agents = agent_stats.head(20)
    
    fig = px.bar(
        top_agents,
        x='sender_id',
        y='acceptance_rate',
        title='Tasa de Aceptación por Agente (Top 20 por Volumen)',
        labels={'sender_id': 'ID del Agente', 'acceptance_rate': 'Tasa de Aceptación (%)'},
        color='acceptance_rate',
        color_continuous_scale='RdYlGn',
        range_color=[0, 100]
    )
    
    # Add target line
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=80,
        x1=19.5,
        y1=80,
        line=dict(color="green", width=2, dash="dot"),
    )
    
    # Save visualization
    fig.write_html(os.path.join(output_dir, "agent_acceptance_rate.html"))
    
    # Action breakdown by agent
    fig2 = px.bar(
        top_agents,
        x='sender_id',
        y=['accepted', 'edited', 'ignored', 'skipped'],
        title='Distribución de Acciones por Agente (Top 20 por Volumen)',
        labels={'sender_id': 'ID del Agente', 'value': 'Cantidad', 'variable': 'Acción'},
        color_discrete_map={
            'accepted': 'green',
            'edited': 'orange',
            'ignored': 'red',
            'skipped': 'gray'
        }
    )
    fig2.write_html(os.path.join(output_dir, "agent_action_breakdown.html"))
    
    # Save agent stats CSV
    agent_stats.to_csv(os.path.join(output_dir, "agent_stats.csv"), index=False)
    
    return agent_stats

def analyze_trends_over_time(df, output_dir="./output"):
    """
    Analyze trends over time for intents and actions
    """
    print("Analyzing trends over time...")
    
    # Group by date and intent
    intent_daily = df.groupby(['date', 'detected_intent']).agg(
        total_messages=('id', 'count'),
        accepted=('suggestion_accepted', 'sum')
    ).reset_index()
    
    # Calculate acceptance rate
    intent_daily['acceptance_rate'] = (intent_daily['accepted'] / intent_daily['total_messages'] * 100).round(2)
    
    # Create intent trend visualization (for top 5 intents by volume)
    top_intents = df['detected_intent'].value_counts().nlargest(5).index.tolist()
    
    intent_trend_data = intent_daily[intent_daily['detected_intent'].isin(top_intents)]
    
    fig = px.line(
        intent_trend_data,
        x='date',
        y='acceptance_rate',
        color='detected_intent',
        title='Tendencia de Aceptación por Intención (Top 5 por Volumen)',
        labels={'date': 'Fecha', 'acceptance_rate': 'Tasa de Aceptación (%)', 'detected_intent': 'Intención Detectada'}
    )
    
    # Add target line
    fig.add_shape(
        type="line",
        x0=intent_trend_data['date'].min(),
        y0=80,
        x1=intent_trend_data['date'].max(),
        y1=80,
        line=dict(color="green", width=2, dash="dot"),
    )
    
    # Save visualization
    fig.write_html(os.path.join(output_dir, "intent_acceptance_trend.html"))
    
    # Analyze weekly trends
    weekly_stats = df.groupby('week_of_year').agg(
        total_messages=('id', 'count'),
        accepted=('suggestion_accepted', 'sum'),
        edited=('suggestion_edited', 'sum'),
        ignored=('suggestion_ignored', 'sum'),
        skipped=('suggestion_skipped', 'sum')
    ).reset_index()
    
    # Calculate acceptance rates
    weekly_stats['acceptance_rate'] = (weekly_stats['accepted'] / weekly_stats['total_messages'] * 100).round(2)
    
    # Create weekly trend visualization
    fig2 = px.line(
        weekly_stats,
        x='week_of_year',
        y='acceptance_rate',
        title='Tasa de Aceptación Semanal',
        labels={'week_of_year': 'Semana del Año', 'acceptance_rate': 'Tasa de Aceptación (%)'},
        markers=True
    )
    
    # Add target line
    fig2.add_shape(
        type="line",
        x0=weekly_stats['week_of_year'].min(),
        y0=80,
        x1=weekly_stats['week_of_year'].max(),
        y1=80,
        line=dict(color="green", width=2, dash="dot"),
    )
    
    # Save visualization
    fig2.write_html(os.path.join(output_dir, "weekly_acceptance_trend.html"))
    
    # Save weekly stats CSV
    weekly_stats.to_csv(os.path.join(output_dir, "weekly_stats.csv"), index=False)
    
    return intent_daily, weekly_stats

def generate_time_analysis_report(daily_stats, weekday_stats, hourly_stats, agent_stats, output_dir="./output"):
    """
    Generate a comprehensive time analysis report
    """
    print("Generating time analysis report...")
    
    # Calculate overall metrics
    overall_acceptance = daily_stats['accepted'].sum() / daily_stats['total_messages'].sum() * 100
    
    # Find best and worst days
    best_day = daily_stats.loc[daily_stats['acceptance_rate'].idxmax()]
    worst_day = daily_stats.loc[daily_stats['acceptance_rate'].idxmin()]
    
    # Find best and worst day of week
    best_dow = weekday_stats.loc[weekday_stats['acceptance_rate'].idxmax()]
    worst_dow = weekday_stats.loc[weekday_stats['acceptance_rate'].idxmin()]
    
    # Find best and worst hours
    best_hour = hourly_stats.loc[hourly_stats['acceptance_rate'].idxmax()]
    worst_hour = hourly_stats.loc[hourly_stats['acceptance_rate'].idxmin()]
    
    # Generate report markdown
    report = f"""# Análisis Temporal de Sugerencias de IA

## Resumen Ejecutivo

Este análisis examina patrones temporales en las sugerencias de IA para soporte al cliente, con un enfoque en la tasa de aceptación a lo largo del tiempo.

**Tasa de aceptación general: {overall_acceptance:.2f}%** (meta: 80%)

## Hallazgos Clave por Tiempo

### Análisis Diario

- **Mejor día:** {best_day['date']} con tasa de aceptación de {best_day['acceptance_rate']}%
- **Peor día:** {worst_day['date']} con tasa de aceptación de {worst_day['acceptance_rate']}%
- La tasa de aceptación muestra una tendencia {'ascendente' if daily_stats['acceptance_rate'].iloc[-1] > daily_stats['acceptance_rate'].iloc[0] else 'descendente'} durante el período analizado.

### Análisis por Día de la Semana

- **Mejor día de la semana:** {best_dow['day_of_week_es']} ({best_dow['acceptance_rate']}%)
- **Peor día de la semana:** {worst_dow['day_of_week_es']} ({worst_dow['acceptance_rate']}%)
- Los días laborables muestran tasas de aceptación {'mejores' if weekday_stats.iloc[:5]['acceptance_rate'].mean() > weekday_stats.iloc[5:]['acceptance_rate'].mean() else 'peores'} que los fines de semana.

### Análisis por Hora

- **Mejor hora del día:** {best_hour['hour_of_day']}:00 ({best_hour['acceptance_rate']}%)
- **Peor hora del día:** {worst_hour['hour_of_day']}:00 ({worst_hour['acceptance_rate']}%)
- Las horas {'matutinas' if hourly_stats.iloc[6:12]['acceptance_rate'].mean() > hourly_stats.iloc[12:18]['acceptance_rate'].mean() else 'vespertinas'} muestran un mejor rendimiento en general.

## Análisis por Agente

- Los agentes muestran una variación considerable en las tasas de aceptación, desde {agent_stats['acceptance_rate'].min():.2f}% hasta {agent_stats['acceptance_rate'].max():.2f}%.
- Hay {len(agent_stats[agent_stats['acceptance_rate'] >= 80])} agentes que cumplen o superan la meta del 80%.
- Los agentes con mayor volumen {'tienen tasas de aceptación más altas' if agent_stats.head(5)['acceptance_rate'].mean() > agent_stats.tail(5)['acceptance_rate'].mean() else 'tienen tasas de aceptación más bajas'} que aquellos con menor volumen.

## Recomendaciones Basadas en Tiempo

1. **Optimización por día de la semana:**
   - Revisar el enfoque para {worst_dow['day_of_week_es']}, que muestra la tasa de aceptación más baja.
   - Estudiar las prácticas utilizadas en {best_dow['day_of_week_es']} para aplicarlas otros días.

2. **Optimización por hora:**
   - Revisar el modelo de respuesta durante las horas {worst_hour['hour_of_day']}:00 a {(worst_hour['hour_of_day'] + 2) % 24}:00.
   - Considerar ajustes específicos para horas pico de volumen.

3. **Estrategia para agentes:**
   - Realizar capacitación específica para agentes con tasas de aceptación por debajo del 40%.
   - Compartir las mejores prácticas de los agentes con mayor tasa de aceptación.

## Próximos Pasos

1. Realizar un análisis más detallado de los patrones por tienda y región.
2. Investigar si existen correlaciones entre el volumen de mensajes y la tasa de aceptación.
3. Analizar si ciertos tipos de intenciones funcionan mejor en momentos específicos del día o días de la semana.
4. Implementar ajustes al modelo basados en patrones temporales identificados.
"""
    
    # Save report
    with open(os.path.join(output_dir, "time_analysis_report.md"), "w") as f:
        f.write(report)
    
    print(f"Time analysis report saved to {os.path.join(output_dir, 'time_analysis_report.md')}")
    
    return report

# Run complete time-based analysis
def run_time_analysis(df, output_dir="./output"):
    """
    Run complete time-based analysis pipeline
    """
    print("Running time-based analysis...")
    
    # Prepare data for time analysis
    df = prepare_time_data(df)
    
    # Analyze by different time granularities
    daily_stats = analyze_by_day(df, output_dir)
    weekday_stats = analyze_by_day_of_week(df, output_dir)
    hourly_stats, daypart_stats = analyze_by_hour(df, output_dir)
    agent_stats = analyze_by_agent(df, output_dir)
    intent_daily, weekly_stats = analyze_trends_over_time(df, output_dir)
    
    # Generate comprehensive report
    report = generate_time_analysis_report(
        daily_stats, weekday_stats, hourly_stats, agent_stats, output_dir
    )
    
    print("Time-based analysis complete!")
    
    return {
        'daily_stats': daily_stats,
        'weekday_stats': weekday_stats,
        'hourly_stats': hourly_stats,
        'daypart_stats': daypart_stats,
        'agent_stats': agent_stats,
        'intent_daily': intent_daily,
        'weekly_stats': weekly_stats,
        'report': report
    }

# Example usage
if __name__ == "__main__":
    # Load your processed data
    df = pd.read_csv("analysis_dataset.csv")
    
    # Run time analysis
    results = run_time_analysis(df, "./output")