import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Ultrafast Spectroscopy Laser Calculator", layout="wide")

# Título do site
st.title("Data analysis panel")

# Criação das abas
tab1, tab2, tab3, tab4 = st.tabs(["About us", "Knife Edge", "Photon flux", "Exciton per dot"])

# Aba 1: Sobre o grupo
with tab1:
    st.header("About the group")
    st.write("""
        Somos um grupo dedicado à análise de dados científicos, com foco em processos ópticos e fotônicos.
        Este painel foi criado para facilitar nossos cálculos e visualizações de forma colaborativa.
    """)

# Inicializações globais de session_state usadas pela aba Knife Edge
if "dados_feixe" not in st.session_state:
    st.session_state.dados_feixe = []
# chave para o campo de texto (controlamos o conteúdo através de st.session_state["input_feixe_value"])
if "input_feixe_value" not in st.session_state:
    st.session_state.input_feixe_value = ""

# Aba 2: Knife Edge
with tab2:
    st.header("Knife edge")
    st.write("Write down the points of your knife edge (position, power)")

    # Layout em duas colunas: formulário à esquerda, gráfico à direita
    col1, col2 = st.columns([1, 2])

    with col1:
        # Campo de entrada controlado: usamos key fixa e sincronizamos com session_state manualmente
        novo_ponto = st.text_input(
            "Novo ponto (ex: 2.1, 0.85)",
            value=st.session_state.input_feixe_value,
            key="input_feixe_field"
        )

        # Sincroniza o valor digitado com a chave usada internamente
        # (isso garante que st.session_state.input_feixe_value reflita o campo)
        if novo_ponto != st.session_state.input_feixe_value:
            st.session_state.input_feixe_value = novo_ponto

        # Botão para adicionar ponto
        if st.button("Adicionar ponto"):
            ponto_texto = st.session_state.input_feixe_value.strip()
            if not ponto_texto:
                st.warning("Digite um ponto antes de enviar.")
            else:
                # validação simples: deve conter exatamente uma vírgula
                if ponto_texto.count(",") != 1:
                    st.warning("Formato inválido. Use: número, número")
                else:
                    partes = [p.strip() for p in ponto_texto.split(",")]
                    try:
                        x = float(partes[0])
                        y = float(partes[1])
                        st.session_state.dados_feixe.append((x, y))
                        # limpa a variável que mantém o valor do campo e também o campo em tela
                        st.session_state.input_feixe_value = ""
                        # força rerun para o campo aparecer limpo imediatamente
                        st.experimental_rerun()
                    except ValueError:
                        st.warning("Formato inválido. Use: número, número")

        # Botão para limpar os dados
        if st.button("🧹 Limpar todos os pontos", key="limpar_feixe"):
            st.session_state.dados_feixe = []
            st.success("Todos os pontos foram removidos!")
            # opcional: limpar também o campo de entrada
            st.session_state.input_feixe_value = ""
            st.experimental_rerun()

        # Tabela minimizada e botão de download
        if st.session_state.dados_feixe:
            with st.expander("📋 Mostrar tabela de pontos"):
                df_feixe = pd.DataFrame(st.session_state.dados_feixe, columns=["Posição", "Intensidade"])
                st.dataframe(df_feixe)

            conteudo_txt = "\n".join([f"{x},{y}" for x, y in st.session_state.dados_feixe])
            st.download_button(
                label="📁 Baixar dados como .txt",
                data=conteudo_txt,
                file_name="dados_knife_edge.txt",
                mime="text/plain"
            )

    with col2:
        # Gráfico scatter com estética refinada (Times New Roman, tamanho 12) e com bordas fechadas
        if st.session_state.dados_feixe:
            df_feixe = pd.DataFrame(st.session_state.dados_feixe, columns=["Posição", "Intensidade"])

            fig = go.Figure()

            # traça pontos e conecta com linha tracejada fechada (se quiser conectar por ordem adicionada)
            fig.add_trace(go.Scatter(
                x=df_feixe["Posição"],
                y=df_feixe["Intensidade"],
                mode='markers+lines',
                line=dict(dash='dash', color='red'),
                marker=dict(size=8, color='red'),
                name='Pontos'
            ))

            fig.update_layout(
                title="Gráfico Knife-Edge",
                xaxis_title="Posição",
                yaxis_title="Intensidade",
                autosize=False,
                width=600,
                height=600,
                margin=dict(l=40, r=40, t=40, b=40),
                font=dict(family="Times New Roman", size=12),
                xaxis=dict(
                    scaleanchor="y",
                    scaleratio=1,
                    fixedrange=True,
                    showline=True,
                    linewidth=2,
                    linecolor='gray',
                    mirror=True,
                    ticks='outside',
                    tickfont=dict(family="Times New Roman", size=12)
                ),
                yaxis=dict(
                    fixedrange=True,
                    showline=True,
                    linewidth=2,
                    linecolor='gray',
                    mirror=True,
                    ticks='outside',
                    tickfont=dict(family="Times New Roman", size=12)
                ),
                dragmode=False
            )

            st.plotly_chart(fig, use_container_width=False)

# Aba 3: Photon flux
with tab3:
    st.header("Photon flux")
    st.write("Set the wavelength and the spot size")

# Aba 4: Exciton per dot
with tab4:
    st.header("Exciton per dot")
    st.write("Set wavelength")
