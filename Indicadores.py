import streamlit as st
import plotly.graph_objects as go

faturamento = {
    'Janeiro': 5000,
    'Fevereiro': 6000,
    'Março': 7000,
    'Abril': 5000,
    'Maio': 6000,
    'Junho': 7000,
    # adicione mais meses aqui
}

# Considerando que o lucro líquido é uma porcentagem do faturamento
lucro_liquido = {mes: valor * 0.1 for mes, valor in faturamento.items()} # Altere 0.1 para a porcentagem de lucro real

max_faturamento = max(faturamento.values())
meses = list(faturamento.keys())

col1, col2, col3 = st.columns(3)

with col2:
    mes_selecionado = st.selectbox('Faturamento mensal selecione mês', meses)
    st.markdown(f'**Faturamento referente a {mes_selecionado}: RS {faturamento[mes_selecionado]}**')

with col3:
    # Criando o gráfico de velocímetro
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = lucro_liquido[mes_selecionado],
        title = {'text': "Lucro Líquido"},
        gauge = {'axis': {'range': [None, max_faturamento * 0.1]}, 'bar': {'color': 'red'}}) # Altere a cor como desejar
    )

    st.plotly_chart(fig, use_container_width=True)
