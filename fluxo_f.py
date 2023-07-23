import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import calendar
import warnings
warnings.filterwarnings("ignore")


#codigo em java h1até h6 mexe no tamanho do titulo
st.markdown("<h6 style='text-align: center; color: black;'>Previsão de Faturamento e Quantidade de Vendas</h3>", unsafe_allow_html=True)

# Evita warnings no output do Streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)


# Definir as datas para o período histórico
start_date = '2020-01-01'
end_date = '2023-08-31'
date_rng = pd.date_range(start=start_date, end=end_date, freq='MS')

# Criar um DataFrame vazio para armazenar os dados de faturamento mensal
df = pd.DataFrame(date_rng, columns=['date'])
df['faturamento_mensal'] = np.nan

# Criar uma tendência polinomial para o faturamento mensal
np.random.seed(42)
tendencia_crescente = 5000
x = np.arange(len(df))
y_tendencia = 0.02 * x**2 + tendencia_crescente * x
df['faturamento_mensal'] = y_tendencia + np.random.normal(loc=0, scale=30000, size=len(df))

# Garantir que não haja valores negativos
df['faturamento_mensal'] = df['faturamento_mensal'].apply(lambda x: max(x, 0))

# Definir as datas para a previsão dos próximos 13 meses
forecast_start_date = '2023-09-01'
forecast_end_date = '2024-09-01'
forecast_date_rng = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='MS')

# Criar um DataFrame para armazenar as previsões dos próximos 13 meses
forecast_df = pd.DataFrame(forecast_date_rng, columns=['date'])

# Criar o modelo de regressão
model = LinearRegression()

# Ajustar o modelo de regressão com os dados históricos
X = df.index.values.reshape(-1, 1)
y = df['faturamento_mensal'].values
model.fit(X, y)

# Fazer a previsão para os próximos 13 meses
forecast_X = np.arange(len(df), len(df) + len(forecast_df)).reshape(-1, 1)
forecast_y = model.predict(forecast_X)

# Garantir que não haja valores negativos nas previsões
forecast_y = np.maximum(forecast_y, 0)

# Preencher o DataFrame de previsão com os valores calculados
forecast_df['faturamento_mensal'] = forecast_y

# Concatenar os DataFrames de histórico e previsão
full_df = pd.concat([df, forecast_df])



# Definindo a semente para obter resultados reproduzíveis
np.random.seed(123)

# Criando um índice de data a partir de 2020-01-01 até 2023-08-31 com frequência mensal
date_range = pd.date_range(start='2020-01-01', end='2023-08-31', freq='MS')

# Gerando valores aleatórios entre 500 e 3000 com uma tendência ascendente mais acentuada
values = np.random.uniform(low=500, high=3000, size=len(date_range)) + np.linspace(0, 2500, len(date_range))

# Criando o DataFrame e renomeando a coluna para "Quantidade de Vendas do Mês"
df = pd.DataFrame(data=values, index=date_range, columns=['Quantidade de Vendas do Mês'])

# Ajustando o modelo de suavização exponencial com trend aditivo e sazonalidade aditiva
model = ExponentialSmoothing(df, trend='add', seasonal='add', seasonal_periods=12)
model_fit = model.fit()

# Fazendo a previsão para o próximo ano
forecast = model_fit.predict(start=pd.to_datetime('2023-09-01'), end=pd.to_datetime('2023-08-31') + pd.DateOffset(months=12))
###############################################################################
# Criando uma figura com Plotly

# plot com Ploty e configuração 


# Cria a figura para 'Quantidade de Vendas do Mês'
#parte de update_layout olhar na documentaçã odo ploty

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df.index, y=df['Quantidade de Vendas do Mês'], mode='lines', name='quantidade de vendas mês'))
fig1.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name='Previsões', line=dict(dash='dash')))
fig1.update_layout(height=600, width=800, autosize=False, 
                   margin=dict(l=20, r=50, b=100, t=100, pad=10), 
                   title_text="Quantidade de Vendas do Mês Previsões",
                   xaxis=dict(title_text="Data", tickformat='%Y-%m', tickangle=90, dtick='M1', range=[start_date, forecast_end_date]),
                   yaxis=dict(title_text="Quantidade de Vendas do Mês"),
                   legend=dict(x=0, y=1, traceorder="normal", font=dict(family="sans-serif", size=10, color="black"),
                               bgcolor="LightSteelBlue", bordercolor="Black", borderwidth=2))


# Cria a figura para 'Faturamento Mensal'
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=full_df['date'], y=full_df['faturamento_mensal'], mode='lines+markers', name='Faturamento Mensal', line=dict(color='rgb(31, 119, 180)', width=2)))
fig2.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['faturamento_mensal'], mode='lines', name='Previsão', line=dict(color='red', width=2), fill='none'))
fig2.update_layout(height=600, width=800, title_text="Faturamento Mensal",
                  xaxis=dict(title_text="Data", tickformat='%Y-%m', tickangle=90, dtick='M1', range=[start_date, forecast_end_date]),
                  yaxis=dict(title_text="Faturamento Mensal", tickformat='.2f', gridcolor='rgba(211, 211, 211, 0.6)'),
                  legend=dict(x=0, y=1, traceorder="normal", font=dict(family="sans-serif", size=10, color="black"),
                              bgcolor="LightSteelBlue", bordercolor="Black", borderwidth=2))

# Cria a figura com os dois gráficos (ambos)
fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Quantidade de Vendas do Mês'], mode='lines', name='quantidade de vendas mês'), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name='Previsões', line=dict(dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=full_df['date'], y=full_df['faturamento_mensal'], mode='lines+markers', name='Faturamento Mensal', line=dict(color='rgb(31, 119, 180)', width=2)), row=2, col=1)
fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['faturamento_mensal'], mode='lines', name='Previsão', line=dict(color='red', width=2), fill='none'), row=2, col=1)
fig.update_layout(height=600, width=800, title_text="Previsões",
                  xaxis=dict(title_text="Data", tickformat='%Y-%m', tickangle=90, dtick='M1', range=[start_date, forecast_end_date]),
                  xaxis2=dict(title_text="Data", tickformat='%Y-%m', tickangle=90, dtick='M1', range=[start_date, forecast_end_date]),
                  yaxis=dict(title_text="Quantidade de Vendas do Mês"),
                  yaxis2=dict(title_text="Faturamento Mensal", tickformat='.2f', gridcolor='rgba(211, 211, 211, 0.6)'),
                  legend=dict(x=0, y=1, traceorder="normal", font=dict(family="sans-serif", size=10, color="black"),
                              bgcolor="LightSteelBlue", bordercolor="Black", borderwidth=2))




##########$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Lucro_liquido = 2300
Total_investido = 2300
Despesas_mesal = 3000
Investimento = 5000

def create_gauge(current_value, min_value, max_value, quadrant_colors, quadrant_text, sensor_text):
    n_quadrants = len(quadrant_colors) - 1
    hand_length = np.sqrt(2) / 4
    hand_angle = np.pi * (1 - (max(min_value, min(max_value, current_value)) - min_value) / (max_value - min_value))

    fig3 = go.Figure(
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
            width=200,
            height=150,
            paper_bgcolor=quadrant_colors[0],
            annotations=[
                go.layout.Annotation(
                    text=f"<b>{sensor_text}:</b><br>R$ :{current_value}",
                    x=0.5, xanchor="center", xref="paper",
                    y=0.25, yanchor="bottom", yref="paper",
                    showarrow=False,
                ),
            #paper_bgcolor="White", # aqui você coloca a cor que deseja ****quadrant_colors[0]***
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
    
    return fig3
##_______________________
# Configurações personalizadas para cada gráfico
# configs = [
#     {"current_value": Lucro_liquido, "min_value": 0, "max_value": 5000, "quadrant_colors": ["#def", "#f25829", "#f2a529", "#eff229", "#85e043", "#2bad4e"], "quadrant_text": ["", "<b style='font-size:10px;'>V high</b>", "<b style='font-size:10px;'>High</b>", "<b style='font-size:10px;'>M</b>", "<b style='font-size:10px;'>L</b>", "<b style='font-size:10px;'>Very l</b>"], "sensor_text": "Lucro Líquido"},
#     {"current_value": Total_investido, "min_value": 0, "max_value": 5000, "quadrant_colors": ["#def", "#f20030", "#f25a00", "#eff200", "#00e043", "#002b4e"], "quadrant_text": ["", "<b style='font-size:10px;'>V high</b>", "<b style='font-size:10px;'>High</b>", "<b style='font-size:10px;'>M</b>", "<b style='font-size:10px;'>L</b>", "<b style='font-size:10px;'>Very l</b>"], "sensor_text": "Total Investido"},
#     {"current_value": Despesas_mesal, "min_value": 0, "max_value": 5000, "quadrant_colors": ["#def", "#f20030", "#f25a00", "#eff200", "#00e043", "#002b4e"], "quadrant_text": ["", "<b style='font-size:10px;'>V high</b>", "<b style='font-size:10px;'>High</b>", "<b style='font-size:10px;'>M</b>", "<b style='font-size:10px;'>L</b>", "<b style='font-size:10px;'>Very l</b>"], "sensor_text": "Despesas Mensais"},
#     {"current_value": Investimento, "min_value": 0, "max_value": 5000, "quadrant_colors": ["#def", "#f20030", "#f25a00", "#eff200", "#00e043", "#002b4e"], "quadrant_text": ["", "<b style='font-size:10px;'>V high</b>", "<b style='font-size:10px;'>High</b>", "<b style='font-size:10px;'>M</b>", "<b style='font-size:10px;'>L</b>", "<b style='font-size:10px;'>Very l</b>"], "sensor_text": "Investimento"},
# ]


# cols = st.columns(4)  # Define a quantidade de colunas

# # Cria um gráfico para cada configuração
# for i, config in enumerate(configs):
#     fig3 = create_gauge(**config)
#     cols[i].plotly_chart(fig3)  # Adicione a figura à coluna i
# Configurações personalizadas para cada gráfico
###############################################
configs = [
    {"current_value": Lucro_liquido, "min_value": 0, "max_value": 5000, "quadrant_colors": ["#ffffff", "#f25829", "#f2a529", "#eff229", "#85e043", "#2bad4e"], "quadrant_text": ["", "<b style='font-size:10px;'>V high</b>", "<b style='font-size:10px;'>High</b>", "<b style='font-size:10px;'>M</b>", "<b style='font-size:10px;'>L</b>", "<b style='font-size:10px;'>Very l</b>"], "sensor_text": "Lucro Líquido"},
    {"current_value": Total_investido, "min_value": 0, "max_value": 5000, "quadrant_colors": ["#ffffff", "#f20030", "#f25a00", "#eff200", "#00e043", "#002b4e"], "quadrant_text": ["", "<b style='font-size:10px;'>V high</b>", "<b style='font-size:10px;'>High</b>", "<b style='font-size:10px;'>M</b>", "<b style='font-size:10px;'>L</b>", "<b style='font-size:10px;'>Very l</b>"], "sensor_text": "Total Investido"},
    {"current_value": Despesas_mesal, "min_value": 0, "max_value": 5000, "quadrant_colors": ["#ffffff", "#f20030", "#f25a00", "#eff200", "#00e043", "#002b4e"], "quadrant_text": ["", "<b style='font-size:10px;'>V high</b>", "<b style='font-size:10px;'>High</b>", "<b style='font-size:10px;'>M</b>", "<b style='font-size:10px;'>L</b>", "<b style='font-size:10px;'>Very l</b>"], "sensor_text": "Despesas Mensais"},
    {"current_value": Investimento, "min_value": 0, "max_value": 5000, "quadrant_colors": ["#ffffff", "#f20030", "#f25a00", "#eff200", "#00e043", "#002b4e"], "quadrant_text": ["", "<b style='font-size:10px;'>V high</b>", "<b style='font-size:10px;'>High</b>", "<b style='font-size:10px;'>M</b>", "<b style='font-size:10px;'>L</b>", "<b style='font-size:10px;'>Very l</b>"], "sensor_text": "Investimento"},
]

option = st.sidebar.selectbox(
    'Qual gráfico você quer mostrar?',
    ('Todos', 'Gráfico de Faturamento', 'Gráfico de Vendas','Grafico velocimetro'))

options_2 = st.sidebar.selectbox(
    'Indicadores?',
     ["KPI's", 'Analises'])

#KPI = st.sidebar.multiselect("Mostrar todos os KPI's", ['KPI 1', 'KPI 2', 'KPI 3'])


cols = st.columns(4)  # Define a quantidade de colunas

# Criar os gráficos de velocímetro e exibi-los
for i ,config in enumerate(configs):
    fig3 = create_gauge(**config)
    cols[i].plotly_chart(fig3)  # Adicione a figura à coluna i

# Mostra o gráfico selecionado
if 'Todos' in option:
    st.plotly_chart(fig)
if 'Gráfico de Faturamento' in option:
    st.plotly_chart(fig2)
if 'Gráfico de Vendas' in option:
    st.plotly_chart(fig1)
if 'Grafico velocímetro' in option:
    st.plotly_chart(fig3)

