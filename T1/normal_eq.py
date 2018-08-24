import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


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
f = 8

feature_X_train = np.asarray([el[f] for el in X_train]).reshape(45849, 1)
feature_X_test = np.asarray([el[f] for el in X_test]).reshape(8091, 1)

feature_X_train_b = np.c_[np.ones((45849, 1)), feature_X_train]
feature_X_test_b = np.c_[np.ones((8091, 1)), feature_X_test]

Theta = np.linalg.inv(feature_X_train_b.T.dot(feature_X_train_b))
Theta = Theta.dot(feature_X_train_b.T)
Theta = Theta.dot(y_train)

predicted = feature_X_test_b.dot(Theta)

print('Coefficients: \n', Theta)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, predicted))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, predicted))

# Plot outputs
plt.scatter(feature_X_test, y_test,  color='black')
plt.plot(feature_X_test, predicted, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

