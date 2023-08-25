import streamlit as st
import numpy as np
import time
import pickle  
import torch   
import pandas as pd                                       
from gensim.models import KeyedVectors
from transformers import BertTokenizer, BertModel
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from function.lstm_preprocessing import (                         
                                clean,         
                                tokin,
                                predict_ml_class,
                                predict_sentence,
                                predict_single_string,
                                LSTMClassifier
                                )

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

stemmer = SnowballStemmer('russian') 
sw = stopwords.words('russian')  

EMBEDDING_DIM = 32
HIDDEN_DIM = 32
SEQ_LEN = 200
VOCAB_SIZE = 196906
EMBEDDING_DIM = 32
wv = KeyedVectors.load("file/wv.wordvectors", mmap='r')

with open('file/vocab_to_int.txt', 'rb') as f:
    vocab_to_int = pickle.load(f)

embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))

for word, i in vocab_to_int.items():
    try:
        embedding_vector = wv[word]
        embedding_matrix[i] = embedding_vector
    except KeyError as e:
        pass

embedding_layer = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))

model = LSTMClassifier(embedding_dim=EMBEDDING_DIM, hidden_size=HIDDEN_DIM, embedding=embedding_layer).to(DEVICE)
model.load_state_dict(torch.load('models/LTSM_model_epoch_7.pt', map_location='cpu'))

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model_BERT = BertModel.from_pretrained("bert-base-multilingual-cased")

loaded_model = pickle.load(open('models/LogReg.pickle', "rb"))

loaded_classifier = pickle.load(open('models/trained_model.pkl', "rb"))
loaded_vectorizer = pickle.load(open('models/vectorizer.pkl', "rb"))

def main():
    st.title("Классификация отзыва на поликлиники")
    user_input = st.text_area("Введите ваш отзыв:", "")
    return user_input

user_input = main()

def predict_lstm(user_input):
    start_time = time.time()
    prediction = predict_sentence(user_input, model, SEQ_LEN, vocab_to_int)
    end_time = time.time()
    return prediction, round((end_time - start_time), 4)

def predict_bert(user_input):
    start_time = time.time()
    prediction = predict_single_string(user_input, model_BERT, loaded_model)
    end_time = time.time()
    return prediction, round((end_time - start_time), 4)

def predict_ML(user_input):
    start_time = time.time()
    prediction = predict_ml_class(user_input, loaded_vectorizer, loaded_classifier)
    end_time = time.time()
    return prediction, round((end_time - start_time), 4)

if user_input:
    prediction_rnn, time_taken_rnn = predict_ML(user_input)
    st.write("### Bag-of-Words + LogReg")
    st.write("Предсказанный класс:", prediction_rnn)
    st.write("Время предсказания:", time_taken_rnn, "сек.")
    prediction_rnn, time_taken_rnn = predict_lstm(user_input)
    st.write("### LSTM модель")
    st.write("Предсказанный класс:", prediction_rnn)
    st.write("Время предсказания:", time_taken_rnn, "сек.")
    prediction_rnn, time_taken_rnn = predict_bert(user_input)
    st.write("### BERT модель + LogReg")
    st.write("Предсказанный класс:", prediction_rnn)
    st.write("Время предсказания:", time_taken_rnn, "сек.")


st.sidebar.image('images/polyclinic.jpeg', use_column_width=True) 
f1_score_classic_ml = 0.87
f1_score_rnn = 0.88
f1_score_bert = 0.83
f1_score_classic_ml_valid = 0.89
f1_score_rnn_valid = 0.92
f1_score_bert_valid = 0.82
# Создание DataFrame для сравнения результатов



st.sidebar.write("### Сравнительная таблица по метрике f1-macro")
results = {
"Модель": ["Классический ML", "LSTM", "BERT-based"],
"train": [f1_score_classic_ml, f1_score_rnn, f1_score_bert],
"valid": [f1_score_classic_ml_valid, f1_score_rnn_valid, f1_score_bert_valid]
}
results_df = pd.DataFrame(results)
st.sidebar.dataframe(results_df)