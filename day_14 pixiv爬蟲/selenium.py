from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from time import sleep
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

from selenium.webdriver.support import expected_conditions as EC
import time 
import pandas as pd
df_url = {"url": []}
                
chrome_options = Options()
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36")
chrome = webdriver.Chrome('chromedriver',options=chrome_options)

url = 'https://accounts.pixiv.net/login'
chrome.get(url)

ID = chrome.find_element(By.CSS_SELECTOR,'#app-mount-point > div > div > div.sc-2oz7me-0.DjdAN > div.sc-anyl02-2.joiqGm > div > div > div > form > fieldset.sc-bn9ph6-0.kJkgq.sc-2o1uwj-3.diUbPW > label > input')
password = chrome.find_element(By.CSS_SELECTOR,'#app-mount-point > div > div > div.sc-2oz7me-0.DjdAN > div.sc-anyl02-2.joiqGm > div > div > div > form > fieldset.sc-bn9ph6-0.kJkgq.sc-2o1uwj-4.hZIeVE > label > input')

ID.send_keys(你的帳號)
password.send_keys(你的密碼)
chrome.find_element(By.CSS_SELECTOR,'#app-mount-point > div > div > div.sc-2oz7me-0.DjdAN > div.sc-anyl02-2.joiqGm > div > div > div > form > button').click()

time.sleep(5)
chrome.get('https://www.pixiv.net/tags/hololive/illustrations?s_mode=s_tag')

for i in range(10000):
    time.sleep(3)
    chrome.execute_script("window.scrollTo(0,15000)")
    
    print('移動',i)
    time.sleep(1)
    soup = BeautifulSoup(chrome.page_source, 'html.parser')
    img = soup.select('div.sc-rp5asc-9.cYUezH > img')
    for i in img:
        df_url['url'].append(i['src'])

    df = pd.DataFrame(df_url)
    df.to_csv("url.csv")
        
    chrome.find_element(By.CSS_SELECTOR,'#root > div.charcoal-token > div > div.sc-1nr368f-2.kBWrTb > div > div.sc-15n9ncy-0.jORshO > div > section:nth-child(3) > div.sc-l7cibp-0.juyBTC > div > nav > a:nth-child(9) > svg').click()
