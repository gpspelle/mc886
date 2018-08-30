import csv
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
import numpy as np

class my_SGDRegressor:


    def __init__(self):
        print("Hey ho, let's go")

    def fit(self, X_train, y_train, max_iter, verbose):

        self.x = np.ones(len(X_train), dtype='Float64')
        self.theta = np.ones(len(X_train), dtype='Float64')
        eta0 = 0.01
        counter = 0
        self.losses = np.zeros(max_iter)

        while True:
            # When this for ends an epoch has ended
            for st in range(len(X_train)):
                counter += 1
                if counter >= max_iter:
                    return self.losses

                self.losses[counter] = (self.theta.dot(self.x) - y_train[st])**2
                for i in range(len(X_train[st])):
                    self.x[i+1] = X_train[st][i]

                acc = self.theta.dot(self.x) - y_train[st]
                for i in range(len(X_train[st])):
                    self.theta[i] = self.theta[i] - (acc*self.x[i]) * eta0 
                
            X_train, y_train = shuffle(X_train, y_train, random_state=random.seed())
        
    def predict(self, X_test):

        predicted = np.empty(len(X_test))
        for i in range(len(X_test)):
            predicted[i] = self.theta.dot(self.x)

        return predicted

X_test = []
X_train = []
y_test = []
y_train = []
#"","carat","cut","color","clarity","depth","table","price","x","y","z"

cut = dict()

cut['Fair'] = 0
cut['Good'] = 1
cut['Very Good'] = 2
cut['Premium'] = 3
cut['Ideal'] = 4

clarity = dict()

clarity['I1'] = 0
clarity['SI2'] = 1
clarity['SI1'] = 2
clarity['VS2'] = 3
clarity['VS1'] = 4
clarity['VVS2'] = 5
clarity['VVS1'] = 6
clarity['IF'] =  7 

color = dict()

color['J'] = 0
color['I'] = 1
color['H'] = 2
color['G'] = 3
color['F'] = 4
color['E'] = 5
color['D'] = 6

with open('diamonds.csv', 'rt') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    cont = 0
    cont2 = 0
    pos = [1, 5, 6, 8, 9, 10] 
    for row in spamreader:
        for i in range(len(row)):
            row[i] = row[i].replace("\"", "")

        row[2] = cut[row[2]] 
        row[3] = color[row[3]]
        row[4] = clarity[row[4]]

        if cont < 45849:
            X_train[cont:cont+1] = [row] 
            y_train[cont:cont+1] = [float(X_train[cont][-4])]

            for x in pos:
                X_train[cont][x] = float(X_train[cont][x])

            del X_train[cont][-4]
            X_train[cont].pop(0)
            cont+=1
        else:
            X_test[cont2:cont2+1] = [row]
            y_test[cont2:cont2+1] = [float(X_test[cont2][-4])]

            for x in pos:
                X_test[cont2][x] = float(X_test[cont2][x])
            del X_test[cont2][-4]
            X_test[cont2].pop(0)
            cont2+=1

    # Feature value from 0 to 8
    #f = 8 

    #feature_X_train = np.asarray([el[f] for el in X_train]).reshape(45849, 1)
    #feature_X_test = np.asarray([el[f] for el in X_test]).reshape(8091, 1)

    max_iter = 100 

    clf = my_SGDRegressor()
    #losses = clf.fit(feature_X_train, y_train, max_iter=max_iter, verbose=1)
    losses = clf.fit(X_train, y_train, max_iter=max_iter, verbose=1)
    #predicted = clf.predict(feature_X_test)
    predicted = clf.predict(X_test)


    print('Coefficients: \n', clf.theta)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, predicted))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, predicted))

    # Plot outputs
    #plt.scatter(feature_X_test, y_test,  color='black')
    #plt.plot(feature_X_test, predicted, color='blue', linewidth=3)
    print(losses)
    print(np.asarray(list(range(max_iter))))
    plt.plot(np.asarray(list(range(max_iter))), losses, color='blue', linewidth=3)

    #plt.ylim(0.8*min(y_test), 1.2*max(y_test))
    plt.xticks(())
    plt.yticks(())
    plt.show()

