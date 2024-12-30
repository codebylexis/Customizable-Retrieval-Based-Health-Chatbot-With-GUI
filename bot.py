import random
import json
import pickle
from google.protobuf import message
import numpy as np

import nltk
from numpy.lib.function_base import append
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intent.json').read())

file = open('words.pkl', 'rb')
words = pickle.load(file)
file.close()

file = open('classes.pkl', 'rb')
classes = pickle.load(file)
file.close()


# words = pickle.loads(open('words.pkl', 'rb'))
# classes = pickle.loads(open('classes.pkl', 'rb'))

model = load_model('chatbotmodel.h5')

def tokenize_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = tokenize_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word == w:
                bag[i] = 1
    
    return np.array(bag)                

def predict_classes(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    Error_Threshold = 0.35
    results = [[i, r] for i, r in enumerate(res) if r > Error_Threshold]

    results.sort(key=lambda x:x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]], 'probability': str(r[1])})  
    return return_list      


def get_response(intent_list, intent_json):
    tag = intent_list[0]['intent']
    list_of_intents = intent_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def app_response(message):
    ints = predict_classes(message)
    res = get_response(ints, intents)  
    return res  

# print('Bot Running')

# while True:
#     message = input('You: ')
#     ints = predict_classes(message)
#     res = get_response(ints, intents)
#     print(f'Health Bot: {res}')