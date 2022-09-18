import json
import requests
import pandas as pd

headers = {'Referer':'https://www.pixiv.net/'}
urls = list(set(pd.read_csv('midterm.csv')['url'].tolist()))

pixiv_url = []
for url in urls:
    idx = url.find('img-master')
    if idx !=-1:
        pixiv_url.append(url[idx:])
    
for page_cnt, p_url in enumerate(pixiv_url):
    url = p_url.split('p0')
    
    cnt = 0
    while(1):
        url = 'https://i.pximg.net/' + url[0] + f'p{cnt}'+ '_master1200.jpg'
        r = requests.get(url, headers=headers)
        print(url)

        if r.status_code != 200:
            break   
            
        with open(f'./holo/{page_cnt}_{cnt}.jpg','wb') as f:
            f.write(r.content)
            
        cnt+=1
            

        
        