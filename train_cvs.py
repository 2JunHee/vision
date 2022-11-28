import cv2 #opencv 라이브러리
import mediapipe as mp #미디어파이프 라이브러리
import numpy as np #넘파이 라이브러리


max_num_hands = 1 #인식할 수 있는 손의 갯수
gesture = {
    0:'ㄱ', 1:'ㄴ', 2:'ㄷ', 3:'ㄹ', 4:'ㅁ', 5:'ㅂ', 6:'ㅅ',
    7:'ㅇ', 8:'ㅈ', 9:'ㅊ', 10:'ㅋ', 11:'ㅌ', 12:'ㅍ', 13:'ㅎ',
    14:'ㅏ', 15:'ㅑ', 16:'ㅓ', 17:'ㅕ', 18:'ㅗ', 19:'ㅛ', 20:'ㅜ', 21:'ㅠ',
    22:'ㅡ', 23:'ㅣ', 24:'ㅐ', 25:'ㅒ', 26:'ㅔ', 27:'ㅖ', 28:'ㅢ', 29:'ㅚ', 30:'ㅟ'
} #31가지의 제스쳐, 제스쳐 데이터는 손가락 landmark의 각도와 각각의 라벨

mp_hands = mp.solutions.hands #mediapipe의 손, 손가락 추적 솔루션
mp_drawing = mp.solutions.drawing_utils #mediapipe의 landmark를 그리기 위한 유틸리티
hands = mp_hands.Hands( #
    max_num_hands = max_num_hands, min_detection_confidence=0.5, min_tracking_confidence=0.5
    #인식할 수 있는 손의 갯수, 최소 인식의 신뢰도, 최소 추적의 신뢰도
)
file = np.genfromtxt('C:/Users/82103/Desktop/project_vision (3)/data/test_1.csv', delimiter=',') #.csv파일을 불러옴
print(file.shape) #파일의 배열 형태

cap = cv2.VideoCapture(0) #카메라 영상 받아옴

def move(event, x,  y, flags, param): #마우스 이벤트 함수 정의
    global data, file #글로벌 변수 생성
    if event == cv2.EVENT_MOUSEWHEEL: #만약 마우스 휠이 움직면
        file=np.vstack((file, data)) #파일의 배열과 data 배열 결합
        print(file.shape) #파일의 배열 형태

cv2.namedWindow("train_window") #train_widow 창 생성
cv2.setMouseCallback("train_window", move) #train_window창에서의 마우스 이벤트

index = 0 #변수 생성 후 초기화

while cap.isOpened(): #cap이 열려있는 동안
    ret, img = cap.read() # cap비디오의 한 프레임씩 읽기
    if not ret: #만약 비어있다면 
        continue #아래 코드 생략
    
    img=cv2.flip(img, 1) #영상의 좌우반전
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #영상을 BGR->RGB
    
    result=hands.process(img) #손 랜드마크 감지 결과 저장

    img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #영상을 RGB->BGR

    if result.multi_hand_landmarks is not None: #result.multi_hand_landmark가 비어있지 않다면
        #multi_hand_landmark 는 손의 21개의 랜드마크 리스트
        for res in result.multi_hand_landmarks: #반복문
            joint=np.zeros((21,3)) # 21*3 행결생성
            for j,lm in enumerate(res.landmark): #반복문
                joint[j]=[lm.x, lm.y, lm.z] # 21개의 랜드마크를 3개의 좌표로 저장
                #각 랜드마크의 x,y,z 좌표를 joint에 저장
            v1=joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # 각 joint의 번호 인덱스 저장
            v2=joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] #각 joint의 번호 인덱스 저장
            v=v2-v1 #각각의 벡데의 각도 계산
            v=v/np.linalg.norm(v,axis=1)[:,np.newaxis] #벡터 정규화
            #두 벡터의 내적값은 cos값
            angle=np.arccos(np.einsum('nt,nt->n', #arccos에 대입하여 두 벡터가 이루는 각 angle변수에 저장
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],  #15개의 각도
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))  #15개의 각도
            
            angle=np.degrees(angle) #angle은 라디안이기 때문에 degree로 변환
            data=np.array([angle], dtype=np.float32) #배열형태로 저장

            data=np.append(data,index) #index번째 클래스의 학습데이터 생성

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS) #손 관절의 landmark그리기

    cv2.imshow('train_window', img) #train_window에 img 출력
    c = cv2.waitKey(1) #입력키 기다림
    if c == 27: #esc를 누르면
        break #브레이크
    elif c == ord('n'): #n을 누르면
        index+=1 #index 1증가
        print(index)
    elif c == ord('b'): #b를 누르면
        index-=1 #index 1감소
    

np.savetxt('C:/Users/82103/Desktop/project_vision (3)/data/test_1.csv',file, delimiter=',') #파일에 저장