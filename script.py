from glove import *
import emoji
import numpy as np

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

