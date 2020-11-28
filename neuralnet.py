#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 19:18:17 2020

@author: sparshtekriwal
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:31:39 2020

@author: sparshtekriwal
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

class NNObject:
    def __init__(self,x, y, a,z,b,y_hat,J):
        self.x = x
        self.y = y
        self.a = a
        self.z = z
        self.b = b
        self.y_hat = y_hat
        self.J = J

def nnForward(alpha, beta, sample):
    x = sample[1:]
    ## one hot encoding y value into a y vector
    y_val = sample[0]
    y = np.zeros(10)
    np.put(y, y_val, 1)
    x = np.insert(x, 0, 1)
    a = np.dot(alpha, x)
    z = sigmoid(a)
    z = np.insert(z, 0, 1)
    b = np.dot(beta, z)
    y_hat = softmax(b)
    J = -np.sum(np.dot(y, np.log(y_hat)))
    return NNObject(x, y, a,z,b,y_hat,J)

def nnBackward(alpha, beta, o, sample):

    dJdb = (o.y_hat - o.y).reshape(-1,1)
    dbdbeta = o.z.reshape(1,-1)
    g_beta = np.dot(dJdb, dbdbeta)
    dbdz = beta[:,1:]
    dJdz = np.dot( np.transpose(dbdz), dJdb )
    dzda = (o.z * (1 - o.z))[1:].reshape(1,-1)
    dJda = dJdz * dzda.reshape(-1,1)
    g_alpha = np.dot(dJda, o.x.reshape(1,-1) )
    return g_alpha, g_beta

def sgd(alpha_initial, beta_initial, train_data, test_data, learning_rate, num_epoch):
    alpha = np.copy(alpha_initial)
    beta = np.copy(beta_initial)
#    with open(metrics_out, 'w') as f:
    mean_cross_entropy_train =0
    mean_cross_entropy_test =0

    mean_cross_entropy_train_list =[]
    mean_cross_entropy_test_list =[]

    for epoch in range(1,num_epoch+1):
        for sample in train_data:
            o = nnForward(alpha, beta, sample)
            g_alpha, g_beta = nnBackward(alpha, beta, o, sample)
            alpha -= learning_rate * g_alpha
            beta -= learning_rate * g_beta

        sum_entropy_train = 0
        for sample in train_data:
            sum_entropy_train += nnForward(alpha, beta, sample).J
        mean_cross_entropy_train = sum_entropy_train / len(train_data)

        sum_entropy_test = 0
        for sample in test_data:
            sum_entropy_test += nnForward(alpha, beta, sample).J
        mean_cross_entropy_test = sum_entropy_test / len(test_data)

        mean_cross_entropy_train_list.append(mean_cross_entropy_train)
        mean_cross_entropy_test_list.append(mean_cross_entropy_test)

    plt.plot(list(range(1,num_epoch+1)), mean_cross_entropy_train_list, label="train" )
    plt.plot(list(range(1,num_epoch+1)), mean_cross_entropy_test_list, label="test" )
    plt.legend( loc='upper right')
    plt.xlabel("Epochs")
    plt.ylabel("Average Cross Entropy")
    plt.title("Learning Rate: "+ str(learning_rate))
    plt.show()
#            f.write("epoch=%s crossentropy(train): %s\nepoch=%s crossentropy(test): %s\n" %(epoch, mean_cross_entropy_train, epoch, mean_cross_entropy_test))
    return alpha, beta, mean_cross_entropy_train, mean_cross_entropy_test


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def softmax(x):
    denominator = np.sum(np.exp(x))
    return np.exp(x)/denominator

def classify(data, alpha, beta):
    labels = []
    for row in data:
        x = row[1:]
        x = np.insert(x, 0, 1)
        a = np.dot(alpha, x)
        z = sigmoid(a)
        z = np.insert(z, 0, 1)
        b = np.dot(beta, z)
        labels.append(np.argmax(softmax(b)))
    return np.array(labels)


# reads data with attribute names
def read_file(file_name):
    return np.genfromtxt(file_name, delimiter=',', dtype=np.int32)

def write_file(file_name, labels):
    with open(file_name, 'w') as f:
        for item in labels:
            f.write("%s\n" % item)


def calculate_error(true_labels, predicted_labels):
    error_count=0
    try:
        for i in range(len(true_labels)):
            if(true_labels[i]!=predicted_labels[i]):
                error_count+=1
    except:
        print("Dimention mismatch while calculating error")
    return error_count/len(true_labels)


if __name__ == '__main__':
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = int(sys.argv[8])
#    learning_rate= float(sys.argv[9])

    train_data = read_file(train_input)
    test_data = read_file(test_input)


    train_entropies =[]
    test_entropies =[]
    hues = ["train","test"] * 5
    for learning_rate in [0.1, 0.01, 0.001]:

        if(init_flag == 2):
            alpha_initial = np.full( (hidden_units, len(train_data[0]) ), 0.0 )
            beta_initial = np.full( (10, hidden_units + 1 ), 0.0 )

        if(init_flag == 1):
            alpha_initial = np.random.uniform(-0.1, 0.1, (hidden_units, len(train_data[0]) ))
            alpha_initial[:,0]=0
            beta_initial = np.random.uniform(-0.1, 0.1, (10, hidden_units + 1 ))
            beta_initial[:,0]=0

        alpha, beta, mean_cross_entropy_train, mean_cross_entropy_test = sgd(alpha_initial, beta_initial, train_data, test_data, learning_rate, num_epoch)
        train_entropies.append(mean_cross_entropy_train)
        test_entropies.append(mean_cross_entropy_test)

   plt.plot(hidden_list, train_entropies, label="train")
   plt.plot(hidden_list, test_entropies, label="test")
   plt.legend( loc='upper right')
   plt.xlabel("No. of Hidden Units")
   plt.ylabel("Average Cross Entropy")
   plt.show()



   # Classify images
   train_pred = classify(train_data, alpha, beta)
   test_pred = classify(test_data, alpha, beta)

   #Write labels to file
   write_file(train_out, train_pred)
   write_file(test_out, test_pred)

   #Calculate and write error metrics
   train_error = calculate_error(train_data[:,0], train_pred)
   test_error = calculate_error(test_data[:,0], test_pred)
   with open(metrics_out, 'a') as f:
       f.write("error(train): %s\nerror(test): %s\n" %(train_error, test_error))

