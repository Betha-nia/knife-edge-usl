import streamlit as st
import pandas as pd

st.set_page_config(page_title="Ultrafast Spectroscopy Laser Calculator", layout="wide")

#Title of the site
st.title("Data analysis panel")

#Creating the tabs
tab1,tab2,tab3,tab4 = st.tabs(["About us","Knife Edge","Photon flux","Exciton per dot"])

#Tab1: About us
with tab1:
    st.header("About the group")
    st.write("""
        Somos um grupo dedicado à análise de dados científicos, com foco em processos ópticos e fotônicos.
        Este painel foi criado para facilitar nossos cálculos e visualizações de forma colaborativa.
    """)

#Tab2: Knife Edge
with tab2:
    st.header("Knife edge")
    st.write("""
        Write down the points of your knife edge (position, power)
    """)

    #Inicialization of the list
    if "dados_feixe" not in st.session_state:
        st.session_state.dados_feixe = []
    
    #User send the point
    novo_ponto = st.text_input("New point (ex: 2.1, 0.85)", key="input_feixe")

    if novo_ponto:
        try:
            x, y = map(float, novo_ponto.split(","))
            st.session_state.dados_feixe.append((x, y))
            st.success(f"Point ({x}, {y}) added!")
            st.session_state.input_feixe = ""  # Limpa o campo
        except:
            st.error("Error! Invalid syntax. Try again.")
    
    # Exib graph
    if st.session_state.dados_feixe:
        import pandas as pd
        df_feixe = pd.DataFrame(st.session_state.dados_feixe, columns=["Position (mm)", "Intensity transmitted (%)"])
        st.write("Data inserted:")
        st.dataframe(df_feixe)

        st.write("Knife-edge Graph")
        st.line_chart(df_feixe.set_index("Position"))

#Tab3: Photon flux
with tab3:
    st.header("Photon flux")
    st.write("""
        Set the wavelength and the spot size
    """)

#Tab4: Exciton per dot
with tab4:
    st.header("Exciton per dot")
    st.write("""
        Set wavelength
    """)

