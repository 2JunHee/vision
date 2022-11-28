import cv2
import mediapipe as mp
import numpy as np
import matplotlib as plt
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

file = np.genfromtxt('C:/Users/82103/Desktop/project_vision (3)/data/train_1.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)

sc = Normalizer()
sc.fit(angle)

file = np.genfromtxt('C:/Users/82103/Desktop/project_vision (3)/data/test_1.csv',delimiter=',')
test_angle = file[:,:-1].astype(np.float32)
test_label = file[:, -1].astype(np.float32)

angle_std = sc.transform(angle)
test_angle_std = sc.transform(test_angle)

knn = KNeighborsClassifier(n_neighbors=10, p=2, metric='minkowski')
knn.fit(angle_std, label)
predy_knn = knn.predict(test_angle_std)

svm = SVC(kernel='rbf', gamma = 'auto', max_iter=1000)
svm.fit(angle_std, label)
predy_svm = svm.predict(test_angle_std)

mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', \
                    solver='sgd', alpha=0.01, batch_size=32, \
                    learning_rate_init=0.1, max_iter=5000000) 

mlp.fit(angle_std, label)
predy_mlp = mlp.predict(test_angle_std)

print("knn : ", predy_knn)
# print("svm : ", predy_svm)
# print("mlp : ", predy_mlp)
# print("groun-truth 라벨 : " ,test_label)

print("knn accuracy : {:.2f}". format(np.mean(predy_knn == test_label)) )
print("svm accuracy : {:.2f}". format(np.mean(predy_svm == test_label)) )
print("mlp accuracy : {:.2f}". format(np.mean(predy_mlp == test_label)) )


