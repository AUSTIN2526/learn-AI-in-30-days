import cv2

cap = cv2.VideoCapture(0)
while(not cap.isOpened()):
    cap = cv2.VideoCapture(0)
    
cnt = 0
classfier = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_alt2.xml")
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faceRects = classfier.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))

    if len(faceRects) > 0:      
        for (x, y, w, h) in faceRects:
            face = frame[y - 10: y + h + 10, x - 10: x + w + 10]
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0,255,0), 2)
            cnt+=1
            cv2.imwrite(f'face/my_face_{cnt}.jpg',face)
    
    if cv2.waitKey(1) == ord('q'):
        break
        
    cv2.imshow('live', frame)


cap.release()
cv2.destroyAllWindows()