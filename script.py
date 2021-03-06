from glove import *
import emoji
import numpy as np
np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
np.random.seed(1)

x_train, y_train = read_csv('data/train.csv')
x_test, y_test = read_csv('data/test.csv')

word_to_vec_map ,word_to_index, index_to_word, = read_glove_vec()

print('Sample Input data:')
for idx in range(10):
    print(x_train[idx], label_to_emoji(y_train[idx]))
              

def word_to_avg(sentence,word_to_vec_map):
    
    word_list = sentence.lower().split()
    
    avg = np.zeros((word_to_vec_map[word_list[0]].shape))
    
    total = 0
    for word in word_list:
        total += word_to_vec_map[word]
    avg = total/len(word_list)
    
    return avg


def model(x,y,word_to_vec_map,learning_rate = 0.01,num_of_iterations = 400):
    
    m = y.shape[0]
    n_y = 5
    n_h = 50
    
    W = np.random.randn(n_y,n_h)/np.sqrt(n_h)
    b = np.zeros((n_y,))
    
    np.random.seed(1)
    
    y_oh = one_hot(y)
    
    for n in range(num_of_iterations):
        for i in range(m):
            avg = word_to_avg(x[i],word_to_vec_map)
            z = np.dot(W,avg)+b
            a = softmax(z)
            
            cost = -np.sum(y_oh[i]*np.log(a))
            
            dz = a - y_oh[i]
            dW = np.dot(dz.reshape(n_y,1), avg.reshape(1, n_h))
            db = dz
            
            W = W - learning_rate * dW
            b = b - learning_rate * db
            
        if n % 100 == 0:
            print("Epoch: " + str(n) + " --- cost = " + str(cost))
            pred = predict(x, y, W, b, word_to_vec_map)
    return pred,W,b

pred, W, b = model(x_train, y_train, word_to_vec_map)
pred_train = predict(x_train, y_train, W, b, word_to_vec_map)
pred_test = predict(x_test, y_test, W, b, word_to_vec_map)

X_my_sentences = np.array(["i adore you", "i love you", "funny lol", "lets play with a ball", "food is ready", "not feeling happy"])
Y_my_labels = np.array([[0], [0], [2], [1], [4],[3]])

pred = predict(X_my_sentences, Y_my_labels , W, b, word_to_vec_map)
print_predictions(X_my_sentences, pred)
print('           '+ label_to_emoji(0)+ '    ' + label_to_emoji(1) + '    ' +  label_to_emoji(2)+ '    ' + label_to_emoji(3)+'   ' + label_to_emoji(4))
print(pd.crosstab(Y_test, pred_test.reshape(56,), rownames=['Actual'], colnames=['Predicted'], margins=True))



def sentences_to_indices(X,word_to_index,max_length):
    
    m=X.shape[0]
    
    X_output = np.zeros((m,max_length))
    
    for i in range(m):
        
        words = X[i].lower().split()
        
        for j,word in enumerate(words):
            
            X_output[i,j] = word_to_index[word]

    return X_output
X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
test = sentences_to_indices(X1,word_to_index,5)

def pretrained_embedding_layer(word_to_vec_map,word_to_index):
    
    vocab_len = len(word_to_index) +1
    emb_dim = word_to_vec_map["apple"].shape[0]
    
    emb_matrix = np.zeros((vocab_len,emb_dim))
    
    for word,idx in word_to_index.items():
        emb_matrix[idx,:] = word_to_vec_map[word]
    
    embedding_layer = Embedding(input_dim = vocab_len, output_dim = emb_dim , trainable =False)
    
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

def lstm_model(input_shape,word_to_vec_map,word_to_index):
    
    sentence_inputs = Input(shape = input_shape ,dtype='int32')
    
    embedding_layer = pretrained_embedding_layer(word_to_vec_map,word_to_index)
    
    embeddings = embedding_layer(sentence_inputs)
    
    X = LSTM(units = 128, return_sequences=True)(embeddings)
    
    X = Dropout(rate=0.5)(X)
    
    X = LSTM(units = 128, return_sequences =False)(X)
    
    X = Dropout(rate=0.5)(X)
    
    X = Dense(units = 5)(X)
    
    X = Activation('softmax')(X)
    
    model = Model(inputs = sentence_inputs,outputs = X)
    
    return model

model = lstm_model((10,),word_to_vec_map, word_to_index)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
X_train_indices = sentences_to_indices(x_train, word_to_index, 10)
Y_train_oh = one_hot(y_train)
model.fit(X_train_indices, Y_train_oh, epochs = 50, batch_size = 32, shuffle=True)

X_test_indices = sentences_to_indices(x_test, word_to_index,10)
Y_test_oh = one_hot(y_test)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print()
print("Test accuracy = ", acc)

