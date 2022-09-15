import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier




def getTopScore(data,text):
    n = len(data)
    while n > 1:
        n-=1
        for i in range(n):        
            if data[i] < data[i+1]:  
                text[i], text[i+1] = text[i+1], text[i]
                data[i], data[i+1] = data[i+1], data[i]
    return text
    
def text2num(data, top):
    result = []
    for i in data:
        tmp = []
        for j in i.split(' '):
            if j in top:
                tmp.append(2)
            else:
                tmp.append(1)
        if len(tmp)<80:
            tmp = tmp + (80-len(tmp))*[0]
        else:
            tmp = tmp[:80]
        result.append(tmp)
        tmp = []
    return result
    
def randomShuffle(x_batch,y_batch,seed=100):
    random.seed(seed)
    random.shuffle(x_batch)
    random.seed(seed)
    random.shuffle(y_batch)
    
    return x_batch,y_batch
    
def classfier(data):
    real, fake = [], []      
    for text, label in zip(data['sms'].values, data['class'].values):
        if label == 'spam':
            fake.append(text.lower())
        else:
            real.append(text.lower())
    return fake,real
def getTfIdfText(fake,real,max_val = 200):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([' '.join(fake),' '.join(real)]).toarray()
    fake_text_top = getTopScore(X[0],vectorizer.get_feature_names())
    real_text_top = getTopScore(X[1],vectorizer.get_feature_names())
    
    return fake_text_top[:max_val],real_text_top[:max_val]
def splitData(data, split_rate=0.8):
    cnt = int(len(data)*split_rate)
    train_data=data[:cnt]
    test_data =data[cnt:]
    
    return train_data,test_data

def train(train_data,train_label):
    
    model_tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=6, random_state=42)
    model_tree.fit(train_data, train_label)
    y_hat_tree = model_tree.predict(train_data)

    model_RF = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
    model_RF.fit(train_data, train_label)
    Y_hat_RF = model_RF.predict(train_data)
    
    model_xgboost = XGBClassifier(n_estimators=100, learning_rate= 0.3)
    model_xgboost.fit(train_data, train_label)
    Y_hat_xg = model_xgboost.predict(train_data)
    
    n=np.size(train_label)
    
    print('Accuarcy decisionTree: {:.2f}％'.format(sum(np.int_(y_hat_tree==train_label))*100./n))
    print('Accuarcy RandomForest: {:.2f}％'.format(sum(np.int_(Y_hat_RF==train_label))*100./n))
    print('Accuarcy XgBoost: {:.2f}％'.format(sum(np.int_(Y_hat_xg==train_label))*100./n))
data = pd.read_csv('SMSSpamCollection.csv',encoding='cp1252')
fake,real = classfier(data)
fake_text_top,real_text_top = getTfIdfText(fake,real)

fake_data = text2num(fake, fake_text_top)
real_data = text2num(real, real_text_top)

f_train,f_test = splitData(fake_data)
r_train,r_test = splitData(real_data)
train_data,train_label = randomShuffle(f_train+r_train,[0 for i in range(len(f_train))]+[1 for i in range(len(r_train))])
test_data,test_label = randomShuffle(f_test + r_test,[0 for i in range(len(f_test))]+[1 for i in range(len(r_test))])
print('訓練:')
train(np.array(train_data),np.array(train_label))
print('測試:')
train(np.array(test_data),np.array(test_label))


 

    
# train the Random Forest and the Naive Bayes Model using training data



                
                