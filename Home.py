import streamlit as st
import pandas as pd
import shap
import pickle


st.set_page_config(
    layout='wide',
    initial_sidebar_state='expanded',
    page_title="Home")


# --- Initialising SessionState ---
if "load_state" not in st.session_state:
    st.session_state.load_state = False

# --- Layout of the Home page ---

st.title("Prêt à dépenser")
st.subheader("Application d'aide à la décision de prêt")

st.write("""Cette application assiste l'agent de crédit dans sa décision d'accorder un prêt à un client.
      Pour ce faire, un algorithme de machine learning est utilisé pour prédire les difficultées d'un client à rembourser le prêt.
      Pour plus de transparence, cette application fournit également des informations pour expliquer l'algorithme et les prédictions""")

col1, col2 = st.columns(2)



# --- Logo ---
with col1:
    st.image("image/logo.png")

# --- Pages description ---
with col2:

    st.write(" ")
    st.write(" ")  # Empty string to center the following text
    st.write(" ")

    st.subheader("Contenu de l'application :")
    st.markdown("""
     Cette application comprend trois pages :
     1) Informations générales sur la base de données et le modèle
     2) Analyse des clients connus
     3) Prédiction du défaut de paiement pour un nouveau client via une API
     """)


# --- Loading data ---

st.subheader("Chargement de l'application : ")


with st.spinner('initialisation...'):  # Show loading status
    @st.cache  # caching to improve performance and save outputs
    def loading_data():
        df_train = pd.read_csv ("data/df_train.csv", sep=";")
        df_new = pd.read_csv ("data/df_test.csv", sep=";")
        # Loading dataframes df_train
        #url = "https://www.dropbox.com/s/4np9xqqh3a2mjsq/df_train.csv.zip?dl=1"
        #df_train = pd.read_csv(url,
            #compression="zip",
           # sep=';',
            #index_col="SK_ID_CURR")
            # Loading dataframes df_train
        #url = "https://www.dropbox.com/s/r1p43l7ad230zjg/df_test.csv.zip?dl=1"
        #df_new = pd.read_csv( url,
          #  compression="zip",
          #  sep=';',
          #  index_col="SK_ID_CURR")
            
        return df_train, df_new

    st.write("1) Chargement data")
    df_train, df_new = loading_data()

    st.write("2) Chargement du modèle")
    model = "model.pkl"
    Credit_clf_final = pickle.load(open(model, 'rb'))

    st.write("3) Chargement de l'explicateur (Shap) ")
    explainer = shap.TreeExplainer(
        Credit_clf_final, df_train.drop(
            columns="TARGET").fillna(0))

    st.write("4) Enregistrement des variables de session")
    st.session_state.df_train = df_train
    st.session_state.df_new = df_new
    st.session_state.Credit_clf_final = Credit_clf_final
    st.session_state.explainer = explainer

    st.success('Fini!')