import streamlit as st
import plotly.graph_objects as go
import numpy as np

Lucro_liquido = 2300
Total_investido = 2300
Despesas_mesal = 3000
Investimento = 5000

def create_gauge(current_value, min_value, max_value, quadrant_colors, quadrant_text, sensor_text):
    n_quadrants = len(quadrant_colors) - 1
    hand_length = np.sqrt(2) / 4
    hand_angle = np.pi * (1 - (max(min_value, min(max_value, current_value)) - min_value) / (max_value - min_value))

    fig = go.Figure(
        data=[
            go.Pie(
                values=[0.5] + (np.ones(n_quadrants) / 2 / n_quadrants).tolist(),
                rotation=90,
                hole=0.5,
                marker_colors=quadrant_colors,
                text=quadrant_text,
                textinfo="text",
                hoverinfo="skip",
            ),
        ],
        layout=go.Layout(
            showlegend=False,
            margin=dict(b=0,t=10,l=10,r=10),
            width=180,
            height=150,
            paper_bgcolor=quadrant_colors[0],
            annotations=[
                go.layout.Annotation(
                    text=f"<b>{sensor_text}:</b><br>R$ :{current_value}",
                    x=0.5, xanchor="center", xref="paper",
                    y=0.25, yanchor="bottom", yref="paper",
                    showarrow=False,
                )
            ],
            shapes=[
                go.layout.Shape(
                    type="circle",
                    x0=0.48, x1=0.52,
                    y0=0.48, y1=0.52,
                    fillcolor="#333",
                    line_color="#333",
                ),
                go.layout.Shape(
                    type="line",
                    x0=0.5, x1=0.5 + hand_length * np.cos(hand_angle),
                    y0=0.5, y1=0.5 + hand_length * np.sin(hand_angle),
                    line=dict(color="#333", width=4)
                )
            ]
        )
    )
    
    return fig

# Configurações personalizadas para cada gráfico
configs = [
    {"current_value": Lucro_liquido, "min_value": 0, "max_value": 5000, "quadrant_colors": ["#def", "#f25829", "#f2a529", "#eff229", "#85e043", "#2bad4e"], "quadrant_text": ["", "<b style='font-size:10px;'>V high</b>", "<b style='font-size:10px;'>High</b>", "<b style='font-size:10px;'>M</b>", "<b style='font-size:10px;'>L</b>", "<b style='font-size:10px;'>Very l</b>"], "sensor_text": "Lucro Líquido"},
    {"current_value": Total_investido, "min_value": 0, "max_value": 5000, "quadrant_colors": ["#def", "#f20030", "#f25a00", "#eff200", "#00e043", "#002b4e"], "quadrant_text": ["", "<b style='font-size:10px;'>V high</b>", "<b style='font-size:10px;'>High</b>", "<b style='font-size:10px;'>M</b>", "<b style='font-size:10px;'>L</b>", "<b style='font-size:10px;'>Very l</b>"], "sensor_text": "Total Investido"},
    {"current_value": Despesas_mesal, "min_value": 0, "max_value": 5000, "quadrant_colors": ["#def", "#f20030", "#f25a00", "#eff200", "#00e043", "#002b4e"], "quadrant_text": ["", "<b style='font-size:10px;'>V high</b>", "<b style='font-size:10px;'>High</b>", "<b style='font-size:10px;'>M</b>", "<b style='font-size:10px;'>L</b>", "<b style='font-size:10px;'>Very l</b>"], "sensor_text": "Despesas Mensais"},
    {"current_value": Investimento, "min_value": 0, "max_value": 5000, "quadrant_colors": ["#def", "#f20030", "#f25a00", "#eff200", "#00e043", "#002b4e"], "quadrant_text": ["", "<b style='font-size:10px;'>V high</b>", "<b style='font-size:10px;'>High</b>", "<b style='font-size:10px;'>M</b>", "<b style='font-size:10px;'>L</b>", "<b style='font-size:10px;'>Very l</b>"], "sensor_text": "Investimento"},
]


cols = st.columns(4)  # Define a quantidade de colunas

# Cria um gráfico para cada configuração
for i, config in enumerate(configs):
    fig = create_gauge(**config)
    cols[i].plotly_chart(fig)  # Adicione a figura à coluna i
