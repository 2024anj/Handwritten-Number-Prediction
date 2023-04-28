import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
data = pd.read_csv('mnist_train_0_4.csv')

train_data = np.array(data) #putting data into array
train_data = train_data.T #transposing data
Y_train = train_data[0] #labels
X_train = train_data[1:785] #784 pixels
X_train = X_train / 255 #normalize

EXAMPLES = len(Y_train)
FRAC = 1/EXAMPLES

#intialize weight and bias matrices
def matrices():
    #rand generates array full of random values between 0-4
    #subtract by .5 to get values between -.5-.5
    W1 = np.random.rand(5, 784) - .5   #5x784
    W2 = np.random.rand(5, 5) - .5     #5x4
    b1 = np.random.rand(5, 1) - .5     #5x1
    b2 = np.random.rand(5, 1) - .5    #5x1
    return W1, b1, W2, b2

def activation(W1, b1, W2, b2, X):
    #hidden layer values
    Z1 = b1 + W1.dot(X)
    #relu activation
    A1 = np.maximum(Z1, 0) #if Z1>0, A1=Z1 but if Z1<0, A1=0
    #output layer values
    Z2 = b2 + W2.dot(A1)
    #formula = 1/1+e^-x
    A2 = 1/(1+np.exp(-Z2))  #sigmoid activation function
    return Z1, A1, Z2, A2

def reluDeriv(Z):
    return Z > 0 #the derivative is 1 if pos. 

def backProp(A1, A2, W1, b1, b2, W2, Z1, X, Y, alpha):
    #take one-hot of labels
    oneHot = np.zeros((EXAMPLES, 5)) #creates new matrix filled w 0s, size:12664x2
    oneHot[np.arange(EXAMPLES), Y] = 1
    oneHot = oneHot.T #transposing
    #output layer
    der_Z2 = (A2 - oneHot)  #ouput activiation - labels
    der_W2 = FRAC * der_Z2.dot(A1.T)#deriv of weight 2
    der_b2 = FRAC * np.sum(der_Z2,1)#deriv bias 2

    #hidden layer
    der_Z1 = W2.T.dot(der_Z2) * reluDeriv(Z1)#take deriv of relu activation
    der_W1 = FRAC * der_Z1.dot(X.T)#deriv weight 1
    der_b1 = FRAC * np.sum(der_Z1,1)#deriv bias 1

    #update weights and biases
    W1 -= alpha * der_W1
    b1 -= alpha * np.reshape(der_b1, (5,1))#deriv bias 1 to 2x1 matrix
    W2 -= alpha * der_W2
    b2 -= alpha * np.reshape(der_b2, (5,1))#deriv bias 2 to 2x1 matrix
    return W1, b1, W2, b2

def gradient(X, Y, alpha, epoch):
    W1, b1, W2, b2 = matrices() #initializing matrices
    #go thru each example and print the accuracy
    for i in range(epoch):
        Z1, A1, Z2, A2 = activation(W1, b1, W2, b2, X)
        W1, b1, W2, b2 = backProp(A1, A2, W1, b1, b2, W2, Z1, X, Y, alpha)
        
        print("Example", i, ':')
        prediction = np.argmax(A2, axis=0)#argmax - returns indices of max values
        print(prediction, Y)
        accuracy = np.sum(prediction == Y)/ EXAMPLES#adds predictions that are correct and divide by #of examples in set
        print('Accuracy: ', accuracy)
 
    return W1, b1, W2, b2

#-------------------------------------------------------
#classify:
def classify(X, W1, b1, W2, b2):
    Z1, A1, Z2, A2 = activation(W1, b1, W2, b2, X)
    return np.argmax(A2, axis=0)
    
def testing(W1, b1, W2, b2, value):
    prediction = classify(X_train[:, value,None], W1, b1, W2, b2)
    prediction = str(prediction)[1:-1] #just removing brackets from output
    print('TESTING:')
    print('Prediction: ', prediction)
    print('Label: ', Y_train[value])

    #plotting:
    image = X_train[:, value]
    image = image.reshape((28,28))*255#reshaping to a 28x28 matrix, multiplying by 255 bc we divided earlier

    plt.title('Number (0-4)')
    plt.imshow(image)
    plt.gray()
    plt.show()

#training
W1, b1, W2, b2 = gradient(X_train, Y_train, .1, 1000) #can change alpha value and epochs

print('--------------------------------------------')
testing(W1, b1, W2, b2, 0) #to test other, change last parameter
