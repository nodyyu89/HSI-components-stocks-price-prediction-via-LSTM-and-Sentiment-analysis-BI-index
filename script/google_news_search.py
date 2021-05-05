# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 09:18:04 2020

@author: hao yu
"""



from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import InvalidArgumentException
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time
import csv
from datetime import datetime, timedelta
import re
import os

SCROLL_PAUSE_TIME = 0.5


today_date_str = datetime.today().strftime('%Y-%m-%d')
current_time = datetime.now().strftime("%H%M%S")
current_path = os.getcwd()
#base_path = os.path.split(os.path.dirname(current_path))
base_path = os.path.dirname(current_path)
data_path = base_path + '\\data\\'
tool_path = base_path + '\\tool\\'
#base_path = r'C:\Userhaoyu\working\investment\python script\google searched data'
#excel = open(base_path+'\\09988\\google_news'+today_date_str+current_time+'.csv', mode='w', errors='ignored',newline='',encoding='utf-8-sig')  # gb18030  utf-8-sig
#google_news_writer = csv.writer(excel, delimiter=',') # , errors='ignored'
#futu_writer.writerow(['poster','date','tag','page_url','stock_id','topic_like_score','topic_unlike_score','like_no','unlike_no','comment'])
#google_news_writer.writerow(['news_time_str','news_source_str','news_title_str','news_preview_str','news_url_str'])

options = webdriver.ChromeOptions()
#driver.set_page_load_timeout(7)
options.add_argument('--ignore-certificate-errors')
options.add_argument('--ignore-ssl-errors')
options.add_argument("--headless") 
#driver = webdriver.Chrome(base_path+'\chromedriver.exe',chrome_options=options)
driver = webdriver.Chrome(tool_path+'\chromedriver.exe',chrome_options=options)


stock_string = """
長和 00001.HK,
中電控股 00002.HK,
香港中華煤氣 00003.HK,
匯豐控股 00005.HK,
電能實業 00006.HK,
恆生銀行 00011.HK,
恆基地產 00012.HK,
新鴻基地產 00016.HK,
新世界發展 00017.HK,
銀河娛樂 00027.HK,
港鐵公司 00066.HK,
恆隆地產 00101.HK,
吉利汽車 00175.HK,
中信股份 00267.HK,
萬洲國際 00288.HK,
中國石油化工股份 00386.HK,
香港交易所 00388.HK,
創科實業 00669.HK,
中國海外發展 00688.HK,
騰訊控股 00700.HK,
中國聯通 00762.HK,
領展房產基金 00823.HK,
中國石油股份 00857.HK,
中國海洋石油 00883.HK,
建設銀行 00939.HK,
中國移動 00941.HK,
長江基建集團 01038.HK,
恆安國際 01044.HK,
石藥集團 01093.HK,
華潤置地 01109.HK,
長實集團 01113.HK,
中國生物製藥 01177.HK,
友邦保險 01299.HK,
工商銀行 01398.HK,
小米集團－Ｗ 01810.HK,
百威亞太 01876.HK,
金沙中國有限公司 01928.HK,
九龍倉置業 01997.HK,
碧桂園 02007.HK,
瑞聲科技 02018.HK,
安踏體育 02020.HK,
藥明生物 02269.HK,
申洲國際 02313.HK,
中國平安 02318.HK,
蒙牛乳業 02319.HK,
舜宇光學科技 02382.HK,
中銀香港 02388.HK,
中國人壽 02628.HK,
交通銀行 03328.HK,
美團－Ｗ 03690.HK,
中國銀行 03988.HK,
阿里巴巴－ＳＷ 09988.HK"""
stock_string_list = re.split(',', stock_string)
stock_string_list = [x.split('\n')[1] for x in stock_string_list]
#stock_string_list = [x[1] for x in stock_string_list]
stock_string_list = [x.split(' ')[0] for x in stock_string_list]

end_date = datetime.strptime("12/21/2018", "%m/%d/%Y")
#end = datetime.strptime("12/21/2018", "%m/%d/%Y")
#futuredate_list = [end_date + timedelta(days=x) for x in range(1,733)]
end_date = datetime.strptime("12/21/2018", "%m/%d/%Y")

date_format = "%m/%d/%Y"
# =============================================================================
# a = datetime.strptime('8/18/2008', date_format)
# b = datetime.strptime('9/26/2008', date_format)
# =============================================================================
delta = datetime.today() - end_date
futuredate_list = [end_date + timedelta(days=x) for x in range(1,delta.days)]
futuredate_list = [x.strftime("%m/%d/%Y") for x in futuredate_list]
futuredate_tuple_list = []
for date in enumerate(futuredate_list):
    if date[0] < len(futuredate_list)-1:
        futuredate_tuple_list.append((date[1],futuredate_list[date[0]+1]))
    
#start_date = '12/23/2020'
#end_date = '12/21/2018'
for date_tuple in futuredate_tuple_list:
    for search_keyword in stock_string_list:
        excel = open(data_path+'\\hang seng\\'+search_keyword+'_google_news'+'-'+today_date_str+'-'+current_time+'.csv', mode='w', errors='ignored',newline='',encoding='utf-8-sig')  # gb18030  utf-8-sig
        google_news_writer = csv.writer(excel, delimiter=',') # , errors='ignored'
        google_news_writer.writerow(['stock','date_tuple','news_time_str','news_source_str','news_title_str','news_preview_str','news_url_str'])
    
        #url = 'https://www.google.com/search?q='+search_keyword+'&tbs=cdr:1,cd_min:'+end_date+',cd_max:'+start_date+'&tbm=nws&sxsrf=ALeKk02Af0ReLRFTmx4VBgzJXcc2PS852g:1608652743353&ei=xxfiX5-RFcPVmAW14YDIAQ&start=20&sa=N&ved=0ahUKEwjfkNa6-uHtAhXDKqYKHbUwABk4ChDy0wMIiAE&biw=1920&bih=913&dpr='+str(page)
        #for page in range(1,30):
        url = 'https://www.google.com/webhp?tbs=cdr:1,cd_min:'+date_tuple[0]+',cd_max:'+date_tuple[1]+'&tbm=nws&sxsrf=ALeKk02Af0ReLRFTmx4VBgzJXcc2PS852g:1608652743353&ei=xxfiX5-RFcPVmAW14YDIAQ&start=20&sa=N&ved=0ahUKEwjfkNa6-uHtAhXDKqYKHbUwABk4ChDy0wMIiAE&biw=1920&bih=913&dpr=1'#+str(page)
        #https://www.google.com/webhp?tbs=cdr:1,cd_min:12/22/2018,cd_max:12/23/2018&tbm=nws&sxsrf=ALeKk02Af0ReLRFTmx4VBgzJXcc2PS852g:1608652743353&ei=xxfiX5-RFcPVmAW14YDIAQ&start=20&sa=N&ved=0ahUKEwjfkNa6-uHtAhXDKqYKHbUwABk4ChDy0wMIiAE&biw=1920&bih=913&dpr=1
        driver.get(url)
        time.sleep(1)
        google_news_input = driver.find_element_by_xpath('/html/body/div[2]/div[2]/form/div[2]/div[1]/div[1]/div/div[2]/input')
        time.sleep(1)
        google_news_input.send_keys(search_keyword)
        time.sleep(0.5)
        google_news_input.send_keys(Keys.ENTER)
        #google_news_input_btn = driver.find_element_by_xpath('/html/body/div[2]/div[2]/form/div[2]/div[1]/div[3]/center/input[1]')
        #google_news_input_btn.click()
        time.sleep(2)
        for page in range(1,30):
            for news_iter in range(1,12):
                print('search_keyword ',search_keyword)
                print('page ', page)
                time.sleep(0.2)
                news_url_str = ''
                news_title_str = ''
                news_source_str = ''
                news_preview_str = ''
                news_time_str = ''
                
                try:
                    news_url = driver.find_element_by_xpath('/html/body/div[7]/div[2]/div[9]/div[1]/div[2]/div/div[2]/div[2]/div/div/div[%s]/g-card/div/div/div[2]/a'%str(news_iter))
                    news_url_str = news_url.get_attribute('href')
                    print('news_url_str',news_url_str)
                except:
                    print('not getting news url')        
                try:
                    news_source = driver.find_element_by_xpath('/html/body/div[7]/div[2]/div[9]/div[1]/div[2]/div/div[2]/div[2]/div/div/div[%s]/g-card/div/div/div[2]/a/div/div[2]/div[1]'%str(news_iter))
                    news_source_str = news_source.text
                    print('news_source_str',news_source_str)
                except:
                    print('not getting news source')
                try:
                    news_title = driver.find_element_by_xpath('/html/body/div[7]/div[2]/div[9]/div[1]/div[2]/div/div[2]/div[2]/div/div/div[%s]/g-card/div/div/div[2]/a/div/div[2]/div[2]'%str(news_iter))
                    news_title_str = news_title.text
                    print('news_title_str',news_title_str)
                except:
                    print('not getting news title')
                try:
                    news_preview = driver.find_element_by_xpath('/html/body/div[7]/div[2]/div[9]/div[1]/div[2]/div/div[2]/div[2]/div/div/div[%s]/g-card/div/div/div[2]/a/div/div[2]/div[3]/div[1]'%str(news_iter))
                    news_preview_str = news_preview.text
                    print('news_preview_str',news_preview_str)
                except:
                    print('not getting news preview')
                try:
                    news_time = driver.find_element_by_xpath('/html/body/div[7]/div[2]/div[9]/div[1]/div[2]/div/div[2]/div[2]/div/div/div[%s]/g-card/div/div/div[2]/a/div/div[2]/div[3]/div[2]'%str(news_iter))
                    news_time_str = news_time.text
                    print('news_time_str',news_time_str)
                except:
                    print('not getting news time')        
                google_news_writer.writerow([search_keyword,str(date_tuple),news_time_str,news_source_str,news_title_str,news_preview_str,news_url_str])
            
            try:
                next_page_button = driver.find_element_by_class_name('fl')
                next_page_button.click()
                #/html/body/div[7]/div[2]/div[9]/div[1]/div[2]/div/div[5]/div[2]/span[1]/div/table/tbody/tr/td[4]/a/span[1]
                #/html/body/div[7]/div[2]/div[9]/div[1]/div[2]/div/div[5]/div[2]/span[1]/div/table/tbody/tr/td[4]/a/span[2]
                #time.sleep(2)
            except:
                print('not loading next page')
        excel.close()
        #d6cvqb


#/html/body/div[7]/div[2]/div[9]/div[1]/div[2]/div/div[2]/div[2]/div/div/div[2]/g-card/div/div/div[2]/a/div/div[2]/div[1]

#/html/body/div[7]/div[2]/div[9]/div[1]/div[2]/div/div[2]/div[2]/div/div/div[1]/g-card/div/div/div[2]/a/div/div[2]/div[2]

#/html/body/div[7]/div[2]/div[9]/div[1]/div[2]/div/div[2]/div[2]/div/div/div[1]/g-card/div/div/div[2]/a/div/div[2]/div[3]/div[1]
#/html/body/div[7]/div[2]/div[9]/div[1]/div[2]/div/div[2]/div[2]/div/div/div[2]/g-card/div/div/div[2]/a/div/div[2]/div[3]/div[1]

