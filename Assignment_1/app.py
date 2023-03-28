import streamlit as st 
from tinydb import TinyDB, Query, where
from sentence_transformers import SentenceTransformer, util
import numpy as np

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

##### ADDD EDA + CHECK SA

db = TinyDB('hotel_reviews_db.json')
reviews = db.table('reviews')

def find_n_largest(list_item, n):
    largest_idxs =[]
    for i in range(n):
        list_item = np.asarray(list_item)
        largest_val_idx = list_item.argmax()
        largest_idxs.append(largest_val_idx)
        list_item = list_item.tolist()
        del list_item[largest_val_idx]
    return largest_idxs

# 1. page-config

st.set_page_config(page_title='REW/REW',
                    page_icon="🛎️",
                    layout='wide'
)

# 2. Page layout - e.g. a title

st.title("REW/REW - the app to review YOUR hotel")

# 3. Multiselect the button 

names =[]
for x in reviews: 
    name = x.get('Hotel_Name')
    if name not in names: 
        names.append(name)

hotel_name = st.selectbox('Select your hotels', names)

col1, col2 = st.columns(2)

with col1:
    subject = st.text_input('Search for a feature of your hotel ', value='') 
    subject_enc = sbert_model.encode(subject)

with col2:
    st.slider('Days since the reviews', min_value=0, max_value=365) 

chosen_revs = reviews.search(where('Hotel_Name') == hotel_name)#Add days to the reviews here ! theoryy

neg_revs = [review.get("Negative_Review") for review in chosen_revs]
pos_revs = [review.get("Positive_Review") for review in chosen_revs]
all_rev_texts = neg_revs + pos_revs
st.write(all_rev_texts)

revs_enc = sbert_model.encode(all_rev_texts)

# for review in chosen_revs:
#     if review.get("Negative_Review") != "No Negative":
#         st.write(review.get("Negative_Review"))
#     if review.get("Positive_Review") != "No Positive":
#         st.write(review.get("Positive_Review"))
        
cos_sim = util.cos_sim(subject_enc, revs_enc)[0]

largest_idxs = find_n_largest(cos_sim, 10)# put it as a variable
#maybe add the sentiment 
#Remove the No Negative

list_to_show = []
for x in largest_idxs:
    to_be_shown = all_rev_texts[x]
    list_to_show.append(to_be_shown)
    
for x in list_to_show:
    st.write(x)