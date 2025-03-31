import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# load dataset
df = pd.read_csv("wine.csv")

# Feature Selection
X = df[['density', 'sulphates', 'residual_sugar']]
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

#question 3
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

training_accuracy =[]
testing_accuracy = []
neighbors = np.arrange(1, 21)

for i in neighbors:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)

    training_accuracy.append(knn.score(X_train,y_train))
    testing_accuracy.append(knn.score(X_test,y_test))

plt.plot(neighbors, training_accuracy, label="Training Accuracy")
plt.plot(neighbors, testing_accuracy, label="Testing Accuracy")
plt.xlabel("Number of neighbors")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()