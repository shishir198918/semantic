from langchain_huggingface import HuggingFaceEmbeddings
from pprint import pprint
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
import sys

os.environ["HF_HOME"]="/media/shishir/Windows/Projects/symantic"

embed=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


document=["""Ravindra Kumar Gupta (born 20 September 1975) is a professor of clinical 
          microbiology at the Cambridge Institute of Therapeutic Immunology and Infectious
           Disease at the University of Cambridge.""",
           """He is a Delhi based Stand Up comic who rose to fame quickly with his new viral video.
            His observational based desi humor, distinct dialect, and his ability to make you burst 
            into gut wrenching laughter out of nowhere has made him one of the most sought after comics
            in Delhi right now.""",
            """Ravi bhusan is not funny"""]

def embed_doc(document):
    return embed.embed_documents(document)

def takking_prompt(query):
    
    return embed.embed_query(query)
    



def calculate_cosine(doc,query):
    cosine=cosine_similarity([query],doc)
    return cosine

def fetch_relevent_line(cosine_values,document):
    max_index=np.argmax(cosine_values)
    return document[max_index]
    



#pprint(embed_doc(document))
#pprint(takking_prompt("funny ravi"))
cosine_values=calculate_cosine(embed_doc(document),takking_prompt(" better and funny ravi"))
pprint(fetch_relevent_line(cosine_values,document))
