import streamlit as st

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
        Knife edge calculator
    """)

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

