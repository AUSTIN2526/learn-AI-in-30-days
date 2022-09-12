from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from time import sleep
import pandas as pd

#設定user agent防止網站鎖IP
chrome_options = Options()
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36")
#指定驅動與導入參數
chrome = webdriver.Chrome('chromedriver',options=chrome_options)
#建立我們資料要的dict
data = {'日期':[],
        '成交股數':[],
        '成交金額':[],
        '開盤價':[],
        '最高價':[],
        '最低價':[],
        '收盤價':[],
        '漲跌價差':[],
        '成交筆數':[]}
        
#設定年月日(2010~2022)
for y in range(2010,2023):
    for m in range(1,13):
        #網址格式為yyyy/mm/dd 不能少一碼所以要補0
        if m <10:
            #m的格式是int所以要轉成str才能作文字的相加
            m = '0'+str(m)
        url = f'https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=html&date={y}{m}01&stockNo=2330'
        #前往網站
        chrome.get(url)
        #獲取網站資料
        soup = BeautifulSoup(chrome.page_source, 'html.parser')
        #透過CSS選擇器找到在tbody裡面所有的tr標籤
        for tr in soup.select('tbody > tr'):
            #將\n透過split()分割
            td = tr.text.split('\n')
            data['日期'].append(td[1])
            data['成交股數'].append(td[2])
            data['成交金額'].append(td[3])
            data['開盤價'].append(td[4])
            data['最高價'].append(td[5])
            data['最低價'].append(td[6])
            data['收盤價'].append(td[7])
            data['漲跌價差'].append(td[8])
            data['成交筆數'].append(td[9])
       
        #防止過度請求網站被鎖定IP
        sleep(10)
        
#dict轉成dataframe
df = pd.DataFrame(data)
#存成csv檔案
df.to_csv("data.csv")

