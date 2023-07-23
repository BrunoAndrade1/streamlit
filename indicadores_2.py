import streamlit as st
import plotly.graph_objects as go
import numpy as np

plot_bgcolor = "#def"
quadrant_colors = [plot_bgcolor, "#f25829", "#f2a529", "#eff229", "#85e043", "#2bad4e"] 
quadrant_text = ["", "<b>Very high</b>", "<b>High</b>", "<b>Medium</b>", "<b>Low</b>", "<b>Very low</b>"]
n_quadrants = len(quadrant_colors) - 1

for i in range(4):
    cols = st.columns(4)  # Define a quantidade de colunas
    for j in range(4):
        current_value = np.random.randint(0, 51)  # Substitua isso pela sua lógica de dados
        min_value = 0
        max_value = 50
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
                width=200,  # Ajuste esses valores para alterar o tamanho dos gráficos
                height=200,  # Ajuste esses valores para alterar o tamanho dos gráficos
                paper_bgcolor=plot_bgcolor,
                annotations=[
                    go.layout.Annotation(
                        text=f"<b>IOT sensor value:</b><br>{current_value} units",
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
        
        cols[j].plotly_chart(fig)  # Adicione a figura à coluna j
