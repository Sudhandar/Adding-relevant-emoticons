# Adding-relevant-emoticons
Adding relevant emoji's to text messages using LSTM model

## Sample data:

- The training data consists of text messages and a relevant emoji attached to it. Here the emoji's are numbered from 0 to 4. 
- The task we have at hand is multi class classification.

A set of sample texts and their emoji's,

```
never talk to me again 😞
I am proud of your achievements 😄
It is the worst day in my life 😞
Miss you so much ❤️
food is life 🍴
I love you mum ❤️
Stop saying bullshit 😞
congratulations on your acceptance 😄

```


## Iteration 1: Training the model using a single layer NN in python

- I trained the model using Stochastic Gradient Descent method due to the limited availablity of the data.
- I trained the model for 400 epochs.
- The metric used was accuracy even though i should have gone with the **F1 score**. Since, it was the initial iteration, I thought of going with the simpler metric to get a high level picture of the model.

The following are the metrics,

```
Training set:
Accuracy: 0.977272727273
Test set:
Accuracy: 0.857142857143
```

- It seems the model is overfitting.

The following is the confusion matrix,

```
Predicted  0.0  1.0  2.0  3.0  4.0  All
Actual                                 
0            6    0    0    1    0    7
1            0    8    0    0    0    8
2            2    0   16    0    0   18
3            1    1    2   12    0   16
4            0    0    1    0    6    7
All          9    9   19   13    6   56

```

 Let's use an alternative model.

