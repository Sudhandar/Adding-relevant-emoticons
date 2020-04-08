import numpy as np
import csv
import emoji
from sklearn.preprocessing import OneHotEncoder 
from sklearn.metrics import accuracy_score
 
def one_hot(data):
    onehotencoder = OneHotEncoder() 
    data = onehotencoder.fit_transform(data.reshape(-1,1)).toarray() 
    return data

def read_glove_vec():
    with open('data\glove.6B.50d.txt','r',encoding='utf-8') as f:
        word_to_vec = {}
        words = set()
        for line in f:
            value = line.split()
            word = value[0]
            vector = np.array(value[1:], dtype=np.float64)
            words.add(word)
            word_to_vec[word] = vector
        
        i = 1
        word_to_index = {}
        index_to_word = {}
        for w in sorted(words):
            word_to_index[w] = i
            index_to_word[i] = w
            i = i+1
    return word_to_vec, word_to_index,index_to_word

def read_csv(filename):
    phrase = []
    emoji = []

    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            phrase.append(row[0])
            emoji.append(row[1])

    X = np.asarray(phrase)
    Y = np.asarray(emoji, dtype=int)

    return X, Y

def softmax(x):
    num = np.exp(x - np.max(x))
    return num / num.sum()

emoji_dictionary = {"0": "\u2764\uFE0F",
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

def label_to_emoji(label):
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)

def predict(X , Y , W, b, word_to_vec_map):
  
    m = X.shape[0]
    pred = np.zeros((m, 1))
    
    for j in range(m):
        
        words = X[j].lower().split()
        
        avg = np.zeros((50,))
        for w in words:
            avg += word_to_vec_map[w]
        avg = avg/len(words)

        Z = np.dot(W, avg) + b
        A = softmax(Z)
        pred[j] = np.argmax(A)
        
    print("Accuracy: "  + str(np.mean((pred[:] == Y.reshape(Y.shape[0],1)[:]))))
    
    return pred

def print_predictions(X, pred):
    for i in range(X.shape[0]):
        print(X[i], label_to_emoji(int(pred[i])))
