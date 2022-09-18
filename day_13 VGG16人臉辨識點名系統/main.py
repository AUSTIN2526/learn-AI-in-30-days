import torchvision as tv
import pandas as pd
import torch
import datetime
import cv2

name = ['myface','other']
model = tv.models.vgg16(pretrained=True).eval()
model.load_state_dict(torch.load('model_weights.pth'))
excel_path = 'attend/' + datetime.datetime.now().strftime('%Y-%m-%d') + '.csv'  
transform = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Resize(224),
    tv.transforms.CenterCrop(224),
    tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]) 
try:
    df = pd.read_csv(excel_path, index_col="ID")
except:
    df = pd.DataFrame([[i,'未簽到'] for i in name],columns=['ID', '簽到日期'])
    df.to_csv(excel_path, encoding='utf_8_sig', index=False)
    df = pd.read_csv(excel_path, index_col="ID")
classfier = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_alt2.xml")


    
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
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0,255,0), 2)
            face = frame[y - 10: y + h + 10, x - 10: x + w + 10]
            face = transform(face)
            result = model(face.unsqueeze(0))
            _,faceID = torch.max(result,1)
            faceID = name[int(faceID[0])]
            cv2.putText(frame,faceID ,(x - 30, y - 30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),2)
            
            if df.loc[faceID]['簽到日期']=='未簽到':
                df.loc[faceID]['簽到日期'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                df.to_csv(excel_path,encoding='utf_8_sig')
                    
    if cv2.waitKey(1) == ord('q'):
        break
    
    system_time = datetime.datetime.now().strftime('%H:%M:%S')
    if system_time =='00:00:00': 
        excel_path = 'attend/' + datetime.datetime.now().strftime('%Y-%m-%d') + '.csv'
        df = pd.DataFrame([[i,'未簽到'] for i in name],columns=['ID', '簽到日期'])
        df.to_csv(excel_path, encoding='utf_8_sig', index=False)
        df = pd.read_csv(excel_path, index_col="ID")
        
    cv2.imshow('live', frame)


cap.release()
cv2.destroyAllWindows()

    