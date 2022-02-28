import pandas as pd
import numpy as np
import re
import pickle
import xgboost as xgb
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
snow = SnowballStemmer(language='english')
from rank_bm25 import BM25Okapi
import gensim
import ast
import nltk


#necessary files
prod_info  = pd.read_csv('products.csv')
corpus = prod_info['prod_info'].apply(str).values

file = prod_info['prod_info'].str.cat(sep=', ')
WORDS = Counter(re.findall(r'\w+',file.lower()))
N = sum(WORDS.values())

#bm25 model1 
with open('model1.pkl', 'rb') as f:
  bm25 = pickle.load(f)
#tfidf for lsi 
with open('tfidf_lsi.pkl', 'rb') as f:
  tfidf = pickle.load(f)

#Load the trained Word2Vec models
search_w2v = gensim.models.Word2Vec.load("search_vec_train.model")
title_w2v = gensim.models.Word2Vec.load("title_vec_train.model")
desc_w2v = gensim.models.Word2Vec.load("desc_vec_train.model")

search_words =  search_w2v.wv.index_to_key
title_words = title_w2v.wv.index_to_key
desc_words = desc_w2v.wv.index_to_key



#BM25 FE
file = open("bm25_FE.txt").read()

bm25_FE = ast.literal_eval(file[14:-1])

#trained xgboost model
model = xgb.Booster()
model.load_model('final.model') 
 





# preprocessing funtion 
def clean_text(sent):
    x = sent

    #punctuations and special chars
    x = re.sub('[^a-zA-Z0-9_ ]',' ', x)
    
    #capital letters seperation
    x = ' '.join(re.findall(r'[A-Z]?[^A-Z\s]+|[A-Z]+', x))

    #lowercase
    x = str(x).lower()
    #html tags
    x = re.sub('<.*?>',' ',x)

    
    #normalize units before removing '.' (in. --> inch)

    string = re.findall('\d+ *in\.{0,1} *',x)
    if len(string)!=0:
        for st in string:
            x = x.replace(st, ' inch ') #since 'in' is a preposition we check if it is follwed by a number
    x = re.sub('(inches)', ' inch ', x)
    string = re.findall('\d+ *cu\.{0,1} *',x)#cubic
    if len(string)!=0:
        for st in string:
            x = x.replace(st, ' cubic ')
    string = re.findall('\d+ *m\.{0,1} *',x)#metre
    if len(string)!=0:
        for st in string:
            x = x.replace(st, ' metre ')
    string = re.findall('\d+ *cm\.{0,1} *',x)#centimetre
    if len(string)!=0:
        for st in string:
            x = x.replace(st, ' centimetre ')
    x = re.sub('( centimetres | centi-metres )',' centimetre ',x) 
    x = re.sub('( metres )',' metre ',x)
    x = re.sub('( ft | fts | feets | foot | foots )','feet',x)
    x = re.sub('( gal | gals | galon )',' gallon ',x)
    x = re.sub('( yds | yd | yards )',' yard ',x)
    x = re.sub('( oz | ozs | ounces | ounc ) ',' ounce ',x)
    x = re.sub('( lb | lbs | pounds )',' pound ',x)
    x = re.sub('( squares | sq )',' square ',x)

    

    #Seperating numbers from letters (8x --> 8 x)
    x = ' '.join([str(ele) for ele in re.split(r'(\d+)', x)])

    #removing extra spaces
    x = re.sub(' +', ' ', x)

    return x




# train_data['product_title'] = train_data['product_title'].apply(clean_text)
# train_data['product_description'] = train_data['product_description'].apply(clean_text)
# train_data['Brand'] = train_data['Brand'].apply(clean_text)

# test_data['product_title'] = test_data['product_title'].apply(clean_text)
# test_data['product_description'] = test_data['product_description'].apply(clean_text)
# test_data['Brand'] = test_data['Brand'].apply(clean_text)




# products_collection_train = train_data[['product_uid','product_title','product_description','Brand']]
# products_collection_test = test_data[['product_uid','product_title','product_description','Brand']]
# products_collection_train['prod_info'] = train_data['product_title'] + " "  + train_data['product_description'] + " " + train_data['Brand'] 
# products_collection_test['prod_info'] = test_data['product_title'] + " "  + test_data['product_description'] + " " + test_data['Brand']
# prod_info = pd.concat([products_collection_train, products_collection_test], axis=0).drop_duplicates()
# # saving the document corpus for spell correction in a text file
# np.savetxt(r'/content/drive/MyDrive/Thesis data/Final model/prod_info.txt', prod_info['prod_info'].values, fmt='%s')
# #saving all the products information in a csv file 
# #Original raw data
# raw_data = pd.concat([pd.read_csv('/content/drive/MyDrive/home-depot-product-search-relevance/train.csv', encoding='latin-1').drop(['id','search_term','relevance'],axis=1), pd.read_csv('/content/drive/MyDrive/home-depot-product-search-relevance/test.csv', encoding='latin-1').drop(['id','search_term'],axis=1)], axis=0).drop_duplicates('product_uid')
# data = pd.merge(raw_data,prod_info , on='product_uid', how='inner').drop_duplicates('product_uid')
# data.to_csv('/content/drive/MyDrive/Thesis data/Final model/products.csv',index = False)



#spell corrector
def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N
def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)
def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or set([word]))
def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)
def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)
def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
def corrected_term(term):
  temp = term.lower().split()
  temp = [correction(word) for word in temp]
  return ' '.join(temp)

# data.head()

class query_processing():

    def __init__(self,WORDS,N):
        self.WORDS = WORDS
        self.N = N


    #spell corrector
    #http://norvig.com/spell-correct.html


    def P(self,word): 
        "Probability of `word`."
        return self.WORDS[word] / self.N
    def correction(self,word): 
        "Most probable spelling correction for word."
        return max(self.candidates(word), key=self.P)
    def candidates(self,word): 
        "Generate possible spelling corrections for word."
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or set([word]))
    def known(self,words): 
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.WORDS)
    def edits1(self,word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)
    def edits2(self,word): 
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))
    def corrected_term(self,term):
        temp = term.lower().split()
        temp = [self.correction(word) for word in temp]
        return ' '.join(temp)

    #cleaning and preprocessing
    # preprocessing funtion 
    def clean_text(self,sent):

        sent = self.corrected_term(sent)
        
        x = sent

        #punctuations and special chars
        x = re.sub('[^a-zA-Z0-9_ ]',' ', x)
        
        #capital letters seperation
        x = ' '.join(re.findall(r'[A-Z]?[^A-Z\s]+|[A-Z]+', x))

        #lowercase
        x = str(x).lower()
        #html tags
        x = re.sub('<.*?>',' ',x)

        
        #normalize units before removing '.' (in. --> inch)

        string = re.findall('\d+ *in\.{0,1} *',x)
        if len(string)!=0:
            for st in string:
                x = x.replace(st, ' inch ') #since 'in' is a preposition we check if it is follwed by a number
        x = re.sub('(inches)', ' inch ', x)
        string = re.findall('\d+ *cu\.{0,1} *',x)#cubic
        if len(string)!=0:
            for st in string:
                x = x.replace(st, ' cubic ')
        string = re.findall('\d+ *m\.{0,1} *',x)#metre
        if len(string)!=0:
            for st in string:
                x = x.replace(st, ' metre ')
        string = re.findall('\d+ *cm\.{0,1} *',x)#centimetre
        if len(string)!=0:
            for st in string:
                x = x.replace(st, ' centimetre ')
        x = re.sub('( centimetres | centi-metres )',' centimetre ',x) 
        x = re.sub('( metres )',' metre ',x)
        x = re.sub('( ft | fts | feets | foot | foots )','feet',x)
        x = re.sub('( gal | gals | galon )',' gallon ',x)
        x = re.sub('( yds | yd | yards )',' yard ',x)
        x = re.sub('( oz | ozs | ounces | ounc ) ',' ounce ',x)
        x = re.sub('( lb | lbs | pounds )',' pound ',x)
        x = re.sub('( squares | sq )',' square ',x)

    

        #Seperating numbers from letters (8x --> 8 x)
        x = ' '.join([str(ele) for ele in re.split(r'(\d+)', x)])

        #removing extra spaces
        x = re.sub(' +', ' ', x)

        return x

    #function to remove stopwords and perform stemming
    def preproecessing(self,sent): 

        sent = self.clean_text(sent)

        sent = sent.replace('_', ' _ ')
        words = sent.split()
        #not including stopwords 
        words = [w for w in words if not w in set(stopwords.words('english'))]
        #coverting to stem form
        words = [snow.stem(word) for word in words]
        return ' '.join(words)


# corpus = data['prod_info'].apply(str).values
# tokens = [doc.split() for doc in corpus]
# bm25 = BM25Okapi(tokens)


# with open('/content/drive/MyDrive/Thesis data/Final model/model1.pkl', 'wb') as f:
#   pickle.dump(bm25, f)


class model1():

    def __init__(self,prod_info,corpus,bm25):
        self.df = prod_info
        self.corpus = corpus
        self.bm25 = bm25
        self.M_prod_info = pd.DataFrame()

    def BM25(self,query):
        query = corrected_term(query)
        query = clean_text(query)
        tokens_query = query.split()

        candidates_M = bm25.get_top_n(tokens_query, self.corpus, n=100) #retrive top 100 products (M=100)
        self.M_prod_info = self.df[self.df['prod_info'].isin(candidates_M)].drop('prod_info', axis=1)
        self.M_prod_info['search_term'] = preprocess.preproecessing(query)

        return self.M_prod_info


class F2():

    def __init__(self,prod_info,bm25_FE):
        self.df = pd.DataFrame()
        self.products = prod_info
        self.bm25_FE = bm25_FE #dict of params

    #len of common words
    def len_common(self,search, doc):
        length =[]
        for i in range(len(search)):
            search_words = set(search[i].split())
            doc_words = set(doc[i].split())
            length.append(len(search_words & doc_words))
        return length
    

    def last_in_doc(self,search, doc):
        last = []
        for i in range(len(search)):
            last_term = search[i].split()[-1]
            if last_term in doc[i].split():
                last.append(1)
            else:
                last.append(0)
        return last



    def ratio(self,len_search, len_common):
        return len_common/len_search
    
    # average Word2Vec
    def avgw2v(self,corpus,model,vocab):
        avg_w2v_vectors = []; # the avg-w2v for each sentence is stored in this list
        for sentence in corpus:
            vector = np.zeros(100)
            cnt_words =0; 
            for word in sentence.split(): 
                if word in vocab:
                    vector += model.wv.__getitem__(word)
                    cnt_words += 1
            if cnt_words != 0:
                vector /= cnt_words
            avg_w2v_vectors.append(vector)
        return avg_w2v_vectors



    def bm25_transform(self,query,doc,avgdl,idf,k,b ,N):
        d = doc.split()
        sum = 0

        for term in query.split():
            tf = d.count(term)
            if term in idf:
                idf_score = idf[term]
            
            else:
                idf_score = np.log(((N + 0.5)/0.5) + 1)

            bm25_score = idf_score * ((tf*(k+1))/(tf + k*(1 - b + (b*(len(d)/avgdl)))))
            sum = sum + bm25_score

        return sum


    def features(self):

        #calculating basic features 
        self.df['len_search'] = self.products['search_term'].apply(str).apply(lambda i : len(i.split()))
        self.df['len_desc'] = self.products['product_description'].apply(str).apply(lambda i : len(i.split()))
        self.df['len_title'] = self.products['product_title_y'].apply(str).apply(lambda i : len(i.split()))
        self.df['len_brand'] = self.products['Brand'].apply(str).apply(lambda i : len(i.split()))

        #len of common words
        #common words length between raw search and product title
        self.df['common_S_T'] = self.len_common(list(self.products.search_term.values), list(self.products.product_title_y.values))
        #common words length between raw search and product desc
        self.df['common_S_D'] = self.len_common(list(self.products.search_term.values), list(self.products.product_description.values))


        #last term search and product title
        self.df['last_in_title'] = self.last_in_doc(list(self.products.search_term.values), list(self.products.product_title_y.values))
        #last term search and product desc
        self.df['last_in_desc'] = self.last_in_doc(list(self.products.search_term.values), list(self.products.product_description.values))

        #last term search and product brand
        self.df['last_in_brand'] = self.last_in_doc(list(self.products.search_term.values), list(self.products.Brand.apply(str).values))


        #ratio search and product title
        self.df['ratio_title'] = self.ratio(np.array(self.df.len_search.values), np.array(self.df['common_S_T'].values))
        #ratio search and product desc
        self.df['ratio_desc'] = self.ratio(np.array(self.df.len_search.values), np.array(self.df['common_S_D'].values))

        #ratio search and product brand
        self.df['ratio_brand'] = self.ratio(np.array(self.df.len_search.values), self.len_common(list(self.products.search_term.values), list(self.products.Brand.apply(str).values)))


        #lsi features
        tfidf_S = tfidf.transform(self.products['search_term'])
        tfidf_T = tfidf.transform(self.products['product_title_y'])
        tfidf_D = tfidf.transform(self.products['product_description'])
        self.df['tfidf_cos_sim_S_T'] = [cosine_similarity(tfidf_S[i], tfidf_T[i])[0][0] for i in range(tfidf_S.shape[0])]
        self.df['tfidf_cos_sim_S_D'] = [cosine_similarity(tfidf_S[i], tfidf_D[i])[0][0] for i in range(tfidf_S.shape[0])]

        return self.df


    #BM25 features
    def bm25_features(self):
        f4 = pd.DataFrame()
        f4['bm25_S_T'] = self.products.apply(lambda i: self.bm25_transform(query = i['search_term'], doc = i['product_title_y'],avgdl = self.bm25_FE['avgdl'],idf = self.bm25_FE['idf'], k = self.bm25_FE['k'],b = self.bm25_FE['b'],N=self.bm25_FE['N']), axis=1)
        f4['bm25_S_D'] = self.products.apply(lambda i: self.bm25_transform(query =i['search_term'],  doc =i['product_description'],avgdl = self.bm25_FE['avgdl'],idf = self.bm25_FE['idf'], k = self.bm25_FE['k'],b = self.bm25_FE['b'],N=self.bm25_FE['N']), axis=1)
        f4['bm25_S_B'] = self.products.apply(lambda i: self.bm25_transform(query =i['search_term'],  doc =str(i['Brand']),avgdl = self.bm25_FE['avgdl'],idf = self.bm25_FE['idf'], k = self.bm25_FE['k'],b = self.bm25_FE['b'],N=self.bm25_FE['N']), axis=1)

        return f4

    #word2vec 
    def w2v(self):
        search_avgw2v = self.avgw2v(self.products['search_term'], search_w2v, search_words)
        title_avgw2v = self.avgw2v(self.products['product_title_y'], title_w2v, title_words)
        desc_avgw2v = self.avgw2v(self.products['product_description'], desc_w2v, desc_words)
  
        w2v_features = np.hstack((search_avgw2v,title_avgw2v,desc_avgw2v))
        return w2v_features
    
    #combining all the features
    def final_features(self):
        return pd.DataFrame(np.hstack([self.products.product_uid.values.reshape(-1,1),self.features(),self.bm25_features(),self.w2v()]))



class model2():
    def __init__(self,model,M_products):
        self.model = model
        self.data = M_products.drop(0,axis = 1)
        self.prod = M_products[0]
    
    def results(self):
    
        self.data['predictions'] = self.model.predict(xgb.DMatrix(self.data))
        top_10_relevance_scores = self.data.sort_values('predictions', ascending=False).iloc[:10]['predictions'] #top 10 relevace scores
        top_10_indices = top_10_relevance_scores.index.to_list()
        products = self.prod.iloc[top_10_indices].values
        return products,top_10_relevance_scores.values


preprocess = query_processing(WORDS,N)
model_1 = model1(prod_info,corpus,bm25)

