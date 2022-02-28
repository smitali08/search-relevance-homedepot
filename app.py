import streamlit as st
import pandas as pd
import numpy as np
import re
import xgboost as xgb
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import pickle
import nltk
import gensim
import ast
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
snow = SnowballStemmer(language='english')
from rank_bm25 import BM25Okapi
nltk.download('stopwords')


# Initial page configuration
st.set_page_config(
    page_title='Home Depot Search Relevance',
    layout="wide",
    initial_sidebar_state="expanded",
 )


from final import *


button_style = 'button_style.css'
#css file for button styles
def button_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
button_css(button_style)



#session state for options
if 'options' not in st.session_state:
    st.session_state.options = 'Demo'

#logo
home_depot_logo = 'https://1000logos.net/wp-content/uploads/2017/02/Home-Depot-Logo.png'

# Sidebar
def app_sidebar():
    with st.sidebar:
        st.image(home_depot_logo, width=60)
        # st.markdown("""<style>.big-font {font-size:20px !important;}</style>""", unsafe_allow_html=True)
        # st.markdown('<p class="big-font">Home Depot Search Relevance</p>', unsafe_allow_html=True)
        st.info('**Home Depot Search Relevance**')

        home_button = st.button("Demo")

        data_button = st.button("About Dataset")
        FE_button = st.button('High Level Architecture')

        # blog_button = st.button('Blogs')


        if home_button:
            st.session_state.options = 'Demo'
        if data_button:
            st.session_state.options = 'About Dataset'
        if FE_button:
            st.session_state.options = 'High Level Architecture'
        # if blog_button:
        #     st.session_state.options = 'Blogs'


def home_page():

    buff, col, buff2 = st.columns([1,4,2])
    col.title('Search for products')
    # col.markdown(f'''<p style="font-family:Courier; color:ED7A33; font-size: 20px;">Search for products</p>''',unsafe_allow_html=True)#getting user search query
    user_input = col.text_input('Eg : water heater, chair,etc')

    if user_input:
        bm25_output = model_1.BM25(user_input) #getting M candidate products

        features = F2(bm25_output,bm25_FE) #generating features
        M_data = features.final_features()

        #ML model
        ml_model = model2(model,M_data)
        product_uids, relevance_scores = ml_model.results()

        products = np.array(prod_info.loc[prod_info['product_uid'].isin(product_uids)]['product_title_x'])

        col1, col2, col3 = st.columns([0.5,5,2])
        # col2.markdown(f'''<font color=ED7A33>Top 10 relevant products</font>''',unsafe_allow_html=True)
        # col3.markdown(f'''<font color=ED7A33>Relevance</font>''',unsafe_allow_html=True)
        col2.subheader('Top 10 relevant products')


        for i in range(1,11):
            col1,col2,col3 = st.columns([0.5,5,2])
            col1.markdown(f'''<font color='white'>{i}</font>''',unsafe_allow_html=True)
            col2.write(products[i-1])
            # col3.write(round(relevance_scores[i-1],2))
    	


def dataset_page():
    st.markdown("<br>", unsafe_allow_html=True)

    """
    ## PROBLEM DEFINITION


    A home improvement retailer, The Home Depot held a competition on Kaggle hoping to improve customer shopping experience.
    The task was to develop a model that accurately predicts the relevance score of search results, hence minimizing human input in the search relevance evaluation.
    
    In addition to predicting the relevance scores, I further used these scores to build a full-fledged search engine.

    --------------------------
    ## DATASET

    The data was provided in four csv files:

    1. train.csv : 
        - id - a unique Id representing (search_term, product_uid) pair
        - product_uid - an id for products
        - product_title - product titles
        - search_term - the search query
        - relevance - the average of the relevance ratings ranging from 1(not relevant) to 3(highly relevant) for a given id
    
    2. test.csv :
        - All the features of train data except for relevance score.

    3. product_description.csv:
        - product_uid - an id for products
        - product_description - verbose descriptions for products
    
    4. attributes.csv:
        - product_uid - an id for products
        - name - an attribute name
        - value - the attribute's value
    --------------------------

    """


def model_page():
    st.header('High Level Architecture')
    st.markdown(
    """
    ------------------------------
    """)

    st.image('https://i.imgur.com/2a7ONaI.png')
    
    st.markdown("<br>", unsafe_allow_html=True)

    """
    

    The main goal of this project is to present top ‘N’ relevant products as accurately as possible to the user with the product of highest relevance score being ranked first, second highest being ranked second and so on. This needs to be done within seconds, so it is not possible to calculate the relevance score for all the products in our database because predicting the relevance score for each product and then ranking all of them will take a significant amount of time which won’t satisfy the business constraint of low latency. On that account, I completed this task in two stages.

    In stage 1, I built a much simpler Information Retrieval model (The BM25 model). This model will retrieve the top ‘M’ (M = 100) products which are most relevant to the given search query. 

    In stage 2, the task was to predict relevance scores ranging from 1 to 3 which was mapped to machine learning as a regression problem since the scores are continuous real values. So, I built a supervised ML regression model which predicts the relevance scores more accurately for the subset (top ‘M’) of products retrieved by our first model and henceforth show the top ‘N’ (N=10) most relevant products to the user in decreasing order of relevance.

    """

# def blog_page():
    # st.markdown("<br>", unsafe_allow_html=True)

    # """

    # ## Part 1 : Machine learning model to predict search relevance

    # The effectiveness of any search engine relies heavily on search relevance. In E-commerce, when customers enter their on the website, the idea of relevance is to .....

    
    # (Read full blog on "link : ")

    # ---------------------

    # ## Part 2 : Extension to search engine and deployment on cloud

    # In this article, I will discuss about how I extended the regression model we built in part 1 to a full fledged search engine and how I integrated it into a webapp ....

    # (Read full blog on "link : ")

    # """



#main function
def main():
    app_sidebar()

    if st.session_state.options == 'About Dataset':
        dataset_page()

    if st.session_state.options == 'High Level Architecture':
        model_page()

    # if st.session_state.options == 'Blogs':
    #     blog_page()

    if st.session_state.options == 'Demo':
        home_page()


 # main()
if __name__ == '__main__':
    main()
