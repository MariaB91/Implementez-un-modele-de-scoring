import pandas as pd
import numpy as np
import streamlit as st
import joblib
import seaborn as sns
import shap
import matplotlib.pyplot as plt
import sklearn
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from joblib import load
import wget


def main() :

    @st.cache_data
    def load_data():
   #### IMPORTATION DES DONNÉES #####
        wget.download( "https://www.kaggle.com/datasets/mariabouchehboun/merged-data-csv")  
        data = pd.read_csv('merged_data.csv', index_col='SK_ID_CURR', encoding ='utf-8')

        
        sample = pd.read_csv('X_sample.csv', index_col='SK_ID_CURR', encoding ='utf-8')
        
        description = pd.read_csv("features_descriptions.csv", 
                                  usecols=['Row', 'Description'], index_col=0, encoding= 'unicode_escape')

        target = data.iloc[:, -1:]

        return data, sample, target, description
    
    #modele
    def load_model():
        clf = joblib.load('lgbmclassifier.joblib')
        return clf



    @st.cache_data
    def load_infos_gen(data):
        lst_infos = [data.shape[0],
                     round(data["AMT_INCOME_TOTAL"].mean(), 2),
                     round(data["AMT_CREDIT"].mean(), 2)]

        nb_credits = lst_infos[0]
        rev_moy = lst_infos[1]
        credits_moy = lst_infos[2]

        targets = data.TARGET.value_counts()

        return nb_credits, rev_moy, credits_moy, targets


    def identite_client(data, id):
        data_client = data[data.index == int(id)]
        return data_client

    @st.cache_data
    def load_age_population(data):
        data_age = round((data["DAYS_BIRTH"]/365), 2)
        return data_age

    @st.cache_data
    def load_income_population(sample):
        df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
        df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
        return df_income

    @st.cache_data
    def load_prediction(sample, id,_clf):
        X=sample.iloc[:, :-1]
        score = clf.predict_proba(X[X.index == int(id)])[:,1]
        return score


    #Loading data……
    data, sample, target, description = load_data()
    id_client = sample.index.values
    clf = load_model()


    #######################################
    # DASHBOARD DESIGN
    #######################################

    #Titles display
    st.title("Bank Loans Dashboard")
    st.subheader("Credit score")


    #Customer ID selection
    st.sidebar.header(" **Database's Summary** ")

    #Loading selectbox
    chk_id = st.sidebar.selectbox("Client ID", id_client)

    #Loading general info
 
    nb_credits, rev_moy, credits_moy, targets = load_infos_gen(data)

    ### Display of information in the sidebar ###
    #  #Number of loans in the sample
    st.sidebar.markdown("<u>Number of loans in the sample :</u>", unsafe_allow_html=True)
    st.sidebar.text(nb_credits)

    #Average income
    st.sidebar.markdown("<u>Average income (USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(rev_moy)

    #AMT CREDIT
    st.sidebar.markdown("<u>Average loan amount (USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(credits_moy)
    
   
    #######################################
    # HOME PAGE - MAIN CONTENT
    #######################################
    #Display Customer ID from Sidebar
    st.write("Customer ID selection :", chk_id)


    #Customer information display : Customer Gender, Age, Family status, Children, …
    st.header(" **Detailed Customer information**  ")

    infos_client = identite_client(data, chk_id)
    st.write(" **Gender :** ", infos_client["CODE_GENDER"].values[0])
    st.write(" **Age :** {:.0f} **ans** ".format(int(infos_client["DAYS_BIRTH"]/365))) # type: ignore
    st.write(" **Family status :** ", infos_client["NAME_FAMILY_STATUS"].values[0])
    st.write(" **Number of children :** {:.0f} ".format(infos_client["CNT_CHILDREN"].values[0]))

    #Age distribution plot
    data_age = load_age_population(data)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data_age, edgecolor = 'k', color="pink")
    ax.axvline(int(infos_client["DAYS_BIRTH"].values / 365), color="green", linestyle='--') # type: ignore
    ax.set(title='Customer age', xlabel='Age(Year)', ylabel='')
    st.pyplot(fig)
    
        
    st.subheader("*Income (USD)*")
    st.write(" **Income total :** {:.0f}".format(infos_client["AMT_INCOME_TOTAL"].values[0]))
    st.write(" **Credit amount :** {:.0f}".format(infos_client["AMT_CREDIT"].values[0]))
    st.write(" **Credit annuities :** {:.0f}".format(infos_client["AMT_ANNUITY"].values[0]))
    st.write(" **Amount of property for credit :** {:.0f}".format(infos_client["AMT_GOODS_PRICE"].values[0]))
        
    #Income distribution plot
    data_income = load_income_population(data)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor = 'k', color="pink")
    ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="green", linestyle='--')
    ax.set(title='Customer income', xlabel='Income (USD)', ylabel='')
    st.pyplot(fig)
        
        
   

    #Customer solvability display
    st.header("**Customer file analysis**")
    prediction = load_prediction(sample, chk_id, clf)
    st.write(" **Default probability :** {:.0f} %".format(round(float(prediction)*100, 2)))



    st.markdown("<u>Customer Data :</u>", unsafe_allow_html=True)
    st.write(identite_client(data, chk_id))

    
    #Feature importance / description
    if st.checkbox("Customer ID {:.0f} feature importance ?".format(chk_id)):
        shap.initjs()
        X = sample.iloc[:, :-1]
        X = X[X.index == chk_id]
        number = st.slider("Pick a number of features…", 0, 10, 2)

        fig, ax = plt.subplots(figsize=(10, 10))
        explainer = shap.TreeExplainer(load_model())
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values[0], X, plot_type ="bar", max_display=number, color_bar=False, plot_size=(5, 5))
        st.pyplot(fig)
        
        if st.checkbox("Need help about feature description ?") :
            list_features = description.index.to_list()
            feature = st.selectbox('Feature checklist…', list_features)
            st.table(description.loc[description.index == feature][:1])
        
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)
            
    


        

if __name__ == '__main__':
    main()
