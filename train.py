import cv2
import mediapipe as mp
import numpy as np
import copy
import pygame as py
from hangul_utils import join_jamos
from PIL import ImageFont, ImageDraw, Image
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

max_num_hands = 1
gesture = {
    0:'ㄱ', 1:'ㄴ', 2:'ㄷ', 3:'ㄹ', 4:'ㅁ', 5:'ㅂ', 6:'ㅅ',
    7:'ㅇ', 8:'ㅈ', 9:'ㅊ', 10:'ㅋ', 11:'ㅌ', 12:'ㅍ', 13:'ㅎ',
    14:'ㅏ', 15:'ㅑ', 16:'ㅓ', 17:'ㅕ', 18:'ㅗ', 19:'ㅛ', 20:'ㅜ', 21:'ㅠ',
    22:'ㅡ', 23:'ㅣ', 24:'ㅐ', 25:'ㅒ', 26:'ㅔ', 27:'ㅖ', 28:'ㅢ', 29:'ㅚ', 30:'ㅟ'
}

mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
hands=mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

file=np.genfromtxt('C:/Users/82103/Desktop/project_vision (3)/data/train_1.csv', delimiter=',')
angle=file[:,:-1].astype(np.float32)
label=file[:, -1].astype(np.float32)

sc=StandardScaler()
sc.fit(angle)
angle_std=sc.transform(angle)

svm=SVC(kernel='rbf', gamma='auto', max_iter=1000) #Classification에 사용되는 SVM모델, 학습 반복횟수 1000번
svm.fit(angle_std, label)

knn=KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(angle_std, label)

mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', \
                    solver='sgd', alpha=0.01, batch_size=32, \
                    learning_rate_init=0.1, max_iter=500)
mlp.fit(angle_std, label)

font=ImageFont.truetype("fonts/gulim.ttc", 20)

cap=cv2.VideoCapture(0)

imageT = np.ones((150, 500), dtype=np.uint8)*255
cv2.line(imageT, (350,0),(350,150), (0), 3)
cv2.line(imageT, (350,75),(500,75), (0), 3)
image_clone=copy.copy(imageT)

rect1=py.Rect(350,0,150,75)
rect2=py.Rect(350,75,150,75)

def click(event, x, y, flags, param):
    global word1, word2
    if event == cv2.EVENT_LBUTTONDOWN:
        if rect1.collidepoint((x, y)):
            print(word1)
            word1=word1+text_mlp
            word2=join_jamos(word1)
        elif rect2.collidepoint((x, y)):
            word1=''
            word2=''

word1=''
word2=''

cv2.namedWindow("imageT")
cv2.setMouseCallback("imageT", click)

while cap.isOpened():
    text=''
   
    ret,img=cap.read()
    if not ret:
        continue

    img=cv2.flip(img, 1)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result=hands.process(img)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    imageT=copy.copy(image_clone)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint=np.zeros((21,3))
            for j,lm in enumerate(res.landmark):
                joint[j]=[lm.x, lm.y, lm.z]

            v1=joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:]
            v2=joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]
            v=v2-v1
            v=v/np.linalg.norm(v,axis=1)[:, np.newaxis]

            angle=np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
            angle=np.degrees(angle)

            data=np.array([angle], dtype=np.float32)
            data_std=sc.transform(data)

            pred_knn=knn.predict(data_std)
            idx_knn=pred_knn[0]

            pred_svm=svm.predict(data_std)
            idx_svm=pred_svm[0]

            pred_mlp=mlp.predict(data_std)
            idx_mlp=pred_mlp[0]

            img=Image.fromarray(img)
            draw=ImageDraw.Draw(img)
            org=(res.landmark[0].x, res.landmark[0].y)
            org_=(res.landmark[0].x, res.landmark[0].y+20)
            org__=(res.landmark[0].x, res.landmark[0].y+40)

            text_knn=gesture[idx_knn].upper()
            text_svm=gesture[idx_svm].upper()
            text_mlp=gesture[idx_mlp].upper() 

            draw.text(org,text_mlp,font=font,fill=(0))
            draw.text(org_,text_knn,font=font,fill=(0))
            draw.text(org__,text_svm,font=font,fill=(0))
            img=np.array(img)

            #mp_drawing.draw_landmarks(img,res,mp_hands.HAND_CONNECTIONS)

    imageT=Image.fromarray(imageT)
    draw=ImageDraw.Draw(imageT)
    org1=(100,75)
    draw.text(org1,word2, font=font,fill=(0))
    imageT=np.array(imageT)

    cv2.imshow("camera", img)
    cv2.imshow("imageT", imageT)
    
    k=cv2.waitKey(1)
    if k==27:
        break

            




