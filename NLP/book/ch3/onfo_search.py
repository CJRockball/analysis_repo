#%% 3.1 Code to populate the documents dictionary

def read_documents():
    f = open('data/CISI.ALL')
    merged = ''
    
    for a_line in f.readlines():
        if a_line.startswith('.'):
            merged += '\n' + a_line.strip()
        else:
            merged += ' ' + a_line.strip()
            
    documents = {}
    
    content = ""
    doc_id = ""
    
    for a_line in merged.split('\n'):
        if a_line.startswith('.I'):
            doc_id = a_line.split(" ")[1].strip()
        elif a_line.startswith(".X"):
            documents[doc_id] = content
            content = ""
            doc_id = ""
        else:
            content += a_line.strip()[3:] + " "
    f.close()
    return documents

documents = read_documents()
print(len(documents))
print(documents.get('1'))

# %% 3.2 Code to populate the queries dictionary

def read_queries():
    f = open('data/CISI.QRY')
    merged = ""
    
    for a_line in f.readlines():
        if a_line.startswith("."):
            merged += '\n' + a_line.strip()
        else: 
            merged += " " + a_line.strip()
        
    queries = ()
    content = ""
    qry_id = ""
    
    for a_line in merged.split('\n'):
        if a_line.startswith('.I'):
            if not content=="":
                queries[qry_id] = content
                content = ""
                qry_id = ""
            qry_id = a_line.split(" ")[1].strip()
        elif a_line.startswith('.W') or a_line.startswith('.T'):
            content += a_line.strip()[3:] + " "
    queries[qry_id] = content
    f.close()
    return queries

queries = read_queries()
print(len(queries))
print(queries.get('1'))


# %% 3.3 Code to populate the mappings dictionary

def read_mappings():
    f = open('data/CISI.REL')

    mappings = {}
    
    for a_line in f.readlines():
        voc = a_line.strip().split()
        key = voc[0].strip()
        current_value = voc[1].strip()
        value = []
        if key in mappings.keys():
            value = mappings.get(key)
        value.append(current_value)
        mappings[key] = value
    f.close()
    return mappings
    
mappings = read_mappings()
print(len(mappings))
print(mappings.keys())
print(mappings.get('1'))


#%% 3.4 Preprocess the data in documetns and queries

import nltk
from nltk import word_tokenize
nltk.download('punkt')

def get_words(text):
    word_list = [word for word in word_tokenize(text.lower())]
    return word_list

doc_words = {}
qry_words = {}

for doc_id in documents.keys():
    doc_words[doc_id] = get_words(documents.get(doc_id))
for qry_id in queries.keys():
    qry_words[qry_id] = get_words(queries.get(qry_id))

print(len(doc_words))
print(doc_words.get('1'))
print(len(doc_words.get('1')))
print(len(qry_words))
print(qry_words.get('1'))
print(len(qry_words.get('1')))


# %% 3.5 SImple Boolean search algorithm

def retrieve_documents(doc_words, query):
    docs = []
    for doc in doc_words.keys():
        found = False
        i = 0
        while i<len(query) and not found:
            word = query[i]
            if word in doc_words.get(doc_id):
                docs.append(doc_id)
                found = True
            else: i+=1
    return docs
docs = retrieve_documents(doc_words, qry_words.get('3'))
print(docs[:100])
print(len(docs))

#%% 3.6 Preprocessing: Stopwords and punctuation marks removal

import nltk
import string 
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.corpus import stopwords

def process(text):
    stoplist = set(stopwords.words('english'))
    word_list = [word for word in word_tokenize(text.lower())
                 if not word in stoplist and not word in string.punctuation]
    return word_list

word_list = process(documents.get("1"))
print(word_list)

#%% 3.7 Preprocessing: Stemming
import nltk
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer

def process(text):
    stoplist = set(stopwords.words('english'))
    st = LancasterStemmer()
    word_list = [st.stem(word) for word in word_tokenize(text.lower())
                 if not word in stoplist and not word in string.punctuation]
    return word_list

word_list = process(documents.get('27'))
print(word_list)
word_list = process('organize', 'organizing', 'organizational', 'organ','organic','organizer')
print(word_list)

#%% 3.8 Code to estimate term frequency in documents and queries

def get_terms(text):
    stoplist = set(stopwords.words('english'))
    terms = ()
    st = LancasterStemmer()
    word_list = [st.stem(word) for word in word_tokenize(text.lower())
                 if not word in stoplist and not word in string.puncuation]
    for word in word_list:
        terms[word] = terms.get(word,0) + 1
    return terms

doc_terms = {}
qry_terms = {}
for doc_id in documents.keys():
    doc_terms[doc_id] = get_terms(documents.get(doc_id))
for qry_id in queries.keys():
    qry_terms[qry_id] = get_terms(queries.get(qry_id))

print(len(doc_terms))
print(doc_terms.get('1'))
print(len(doc_terms.get('1')))
print(len(qry_terms))
print(qry_terms.get('1'))
print(len(qry_terms.get('1')))

#%% 3.9 Code to represent the data in a shared space

def collect_vocabulare():
    all_terms = []
    for doc_id in doc_terms.keys():
        for term in doc_terms.get(doc_id).keys():
            all_terms.append(term)
    for qry_id in qry_terms.keys():
        for term in qry_terms.get(qry_id).keys():
            all_terms.append(term)
    return sorted(set(all_terms))

all_terms = collect_vocabulare()
print(len(all_terms))
print(all_terms[:10])

def vectorize(input_features, vocabulary):
    output = []
    for item_id in input_features.keys():
        features = input_features.get(item_id)
        output_vector = []
        for word in vocabulary:
            if word in features.keys():
                output_vector.append(int(features.get(word)))
            else:
                output_vector.append(0)
        output[item_id] = output_vector
    return output

doc_vectors = vectorize(doc_terms, all_terms)
qry_vectors = vectorize(qry_terms, all_terms)

print(len(doc_vectors))
print(len(doc_vectors.get('1460')))
print(len(qry_vectors))
print(len(qry_vectors.get('112')))


#%% 3.10 Code to calculate and apply inverse document frequency weighting

import math

def calculate_idfs(vocabulary, doc_features):
    doc_idfs = {}
    for term in vocabulary:
        doc_count = 0
        for doc_id in doc_features.keys():
            terms = doc_features.get(doc_id)
            if term in terms.keys():
                doc_count += 1 
            doc_idfs[term] = math.log(float(len(doc_features.keys())) / float(1 + doc_count),10)
    return doc_idfs

doc_idfs = calculate_idfs(all_terms, doc_terms)
print(len(doc_idfs))
print(doc_idfs.get('system'))


def vectorize_idf(input_terms, input_idfs, vocabulary):
    output = {}
    for item_id in input_terms.keys():
        terms = input_terms.get(item_id)
        output_vector = []
        for term in vocabulary:
            if term in terms.keys():
                output_vector.append(input_idfs.get(term)*float(terms.get(term)))
            else:
                output_vector.append(float(0))
        output[item_id] = output_vector
    return output

doc_vectors = vectorize_idf(doc_terms, doc_idfs, all_terms)

print(len(doc_vectors))
print(len(doc_vectors.get('1460')))


#%% 3.11 Code to run search algorithm for a given query on the set of documents
from operator import itemgetter

def length(vector):
    sq_length = 0
    for index in range(0, len(vector)):
        sq_length += math.pow(vector[index], 2)
    return math.sqrt(sq_length)
        
def dot_product(vector1, vector2):
    if len(vector1)==len(vector2):
        dot_prod = 0
        for index in range(0, len(vector1)):
            if not vector1[index]==0 and not vector2[index]==0:
                dot_prod += vector1[index]*vector2[index]
        return dot_prod
    else:
        return 'Unmatching dimensionality'
            
def calculate_cosine(query, document):
    cosine = dot_product(query, document) / (length(query)*length(document))
    return cosine

query = qry_vectors.get("3")
results = {}

for doc_id in doc_vectors.key():
    document = doc_vectors.get(doc_id)
    cosine = calculate_cosine(query, document)
    results[doc_id] = cosine

for items in sorted(results.items(), key=itemgetter(1), reverse=True)[:44]:
    print(items[0])
    

#%% 3.12 Estimate precision@k and ratio of vases with at least one relevant document

def calculate_precision(model_output, gold_standard):
    true_pos = 0
    for item in model_output:
        if item in gold_standard:
            true_pos += 1
    return float(true_pos/float(len(model_output)))

def calculate_found(model_output, gold_standard):
    found = 0
    for item in model_output:
        if item in gold_standard:
            found = 1
    return float(found)

precision_all = 0.0
found_all = 0.0
for query_id in mappings.keys():
    gold_standard = mappings.get(str(query_id))
    query = qry_vectors.get(str(query_id))
    results = {}
    model_output = []
    for doc_id in doc_vectors.keys():
        document - doc_vectors.get(doc_id)
        cosine = calculate_cosine(query, document)
        results[doc_id] = cosine
    for items in sorted(results.items(), key=itemgetter(1), reverse=True):
        model_output.append(items[0])
    precision = calculate_precision(model_output, gold_standard)
    found = calculate_found(model_output, gold_standard)
    print(f'{str(query_id)}: {str(precision)}')
    precision_all += precision
    found_all += found
    
print(precision_all/float(len(mappings.keys())))
print(found_all/float(len(mappings.keys())))


#%% 3.13 Code to estimate mean reciprocal rank

rank_all = 0.0
for query_id in mappings.keys():
    gold_standard = mappings.get(str(query_id))
    query = qry_vectors.get(str(query_id))
    results = {}
    for doc_id in doc_vectors.keys():
        document = doc_vectors.get(doc_id)
        cosine = calculate_cosine(query, document)
        results[doc_id] = cosine
    sorted_results = sorted(results.items(), key=itemgetter[1], reverse=True)
    index = 0
    found = False
    while found == False:
        item = sorted_results[index]
        index += 1
        if index == len(sorted_results):
            found = True
        if item[0] in gold_standard:
            found = True
            print(f"{str(query_id)}: {str(float(1)/ float(index))}")
            rank_all += float(1) / float(index)

print(rank_all/float(len(mappings.keys())))







