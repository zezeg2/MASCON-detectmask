from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
import cv2
import datetime

#마스크 착용 안내음성 파일 지정
mixer.init()
sound_nm = mixer.Sound('nomask_case.mp3')
sound_wm = mixer.Sound('wrmask_case.mp3')
prev, switch = 0, 0

now = datetime.datetime.now()
now = now.strftime('%Y-%m-%d-%H-%M')

facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
model = load_model('models'
                   '/test3.model')
#sample image 내에 있는 영상파일로 테스트 가능
#cap = cv2.VideoCapture('sample video/04.mp4')
cap = cv2.VideoCapture('http://whdgus003.iptime.org:8090/?action=stream')
ret, img = cap.read()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./output/'+ now +'.avi', fourcc, cap.get(cv2.CAP_PROP_FPS), (img.shape[1], img.shape[0]))

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    h, w = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
    facenet.setInput(blob)
    dets = facenet.forward()

    result_img = img.copy()

    for i in range(dets.shape[2]):
        confidence = dets[0, 0, i, 2]
        if confidence < 0.5:
            continue

        x1 = int(dets[0, 0, i, 3] * w)
        y1 = int(dets[0, 0, i, 4] * h)
        x2 = int(dets[0, 0, i, 5] * w)
        y2 = int(dets[0, 0, i, 6] * h)

        face = img[y1:y2, x1:x2]


        face_input = cv2.resize(face, dsize=(224, 224))
        face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
        face_input = preprocess_input(face_input)
        face_input = np.expand_dims(face_input, axis=0)

        
        mask, nomask, wrmask= model.predict(face_input).squeeze()

        if max(mask, nomask, wrmask) == mask:
            color = (0, 255, 0)
            label = 'Mask %d%%' % (mask * 100)
            switch = 0
        elif max(mask, nomask, wrmask) == nomask:
            color = (0, 0, 255)
            label = 'No Mask %d%%' % (nomask * 100)
            switch = 1
        else:
            if abs(wrmask - mask) < 0.2:
                color = (0, 255, 0)
                label = 'Mask %d%%' % (mask * 100)
            if abs(wrmask - nomask) < 0.2:
                color = (0, 0, 255)
                label = 'No Mask %d%%' % (nomask * 100)
                switch = 1
            else:
                color = (255, 0, 0)
                label = 'Wrong Mask %d%%' % (wrmask * 100)
                switch = 2
        if not mixer.get_busy():
            if prev < switch:
                if prev + 1 == switch:
                    sound_nm.play()
                if prev + 2 == switch:
                    sound_wm.play()


        cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
        cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                    color=color, thickness=2, lineType=cv2.LINE_AA)

    out.write(result_img)
    cv2.imshow('result', result_img)
    if cv2.waitKey(40) == ord('q'):
        break

out.release()
cap.release()
