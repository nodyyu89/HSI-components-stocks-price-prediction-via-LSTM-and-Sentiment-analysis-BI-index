# -*- coding: utf-8 -*-
"""
Created on Thu May  6 01:01:48 2021

@author: hao yu
"""



import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle
from datetime import datetime, timedelta, date
import re
from hanziconv import HanziConv
from futu import *
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def emotion_analyse(text,pos_words_list,neg_words_list):
    pos_word_count = 0
    neg_word_count = 0
    simiplited_chinese_text = HanziConv.toSimplified(text)
    print('simiplited_chinese_text', simiplited_chinese_text)
    for pos_word_str in pos_words_list:
        if pos_word_str in simiplited_chinese_text:
            pos_word_count += 1
    for neg_word_str in neg_words_list:
        if neg_word_str in simiplited_chinese_text:
            neg_word_count += 1
    polar = 0
    if pos_word_count>neg_word_count:
        polar = 1
    else:
        polar = -1
    return polar           


def BI_func(stock_news_data_by_date):
    pos = 0
    neg = 0
    for sensitive_value in stock_news_data_by_date:
        if sensitive_value['senitive_polarity'] == 1:   #for pos_word_str in pos_words_list:
            pos+=1
        if sensitive_value['senitive_polarity'] == -1:
            neg+=1
    print('pos ',pos)
    print('neg ',neg)
    bi = np.log(1.0 * (1+pos) / (1+neg))
    return bi

def read_stock_news(file_path):
    df = pd.read_csv(file_path, encoding='utf8')
    df_droped = df.dropna()
    df_droped = df_droped.drop_duplicates()
    df_droped = df_droped.values.tolist()
    df_droped_dict_list = []
    for news in df_droped:
        df_droped_dict = {}
        df_droped_dict['search_keyword'] = news[0]
        df_droped_dict['news_time_str'] = news[1]
        df_droped_dict['news_source_str'] = news[2]
        df_droped_dict['news_title_str'] = news[3]
        df_droped_dict['news_preview_str'] = news[4]
        df_droped_dict['news_url_str'] = news[5]
        df_droped_dict['senitive_polarity'] = ''
        df_droped_dict_list.append(df_droped_dict)
    return df_droped_dict_list

def calculate_BI_index(stock_news_data,date_tuple_list):
    stock_news_data_by_date_with_BI = []
    for dates in date_tuple_list:
        print('calculating bi index of ',dates)
        stock_news_data_by_date = []
        for news_ele in stock_news_data:
            if dates in news_ele['news_time_str']:
                stock_news_data_by_date.append(news_ele)       
        for news_with_date in enumerate(stock_news_data_by_date):
            news_str_for_sensitive = news_with_date[1]['news_title_str']+news_with_date[1]['news_preview_str']
            if news_with_date[1]['search_keyword'] in news_str_for_sensitive:
                stock_news_data_by_date[news_with_date[0]]['senitive_polarity'] = emotion_analyse(news_str_for_sensitive
                                                                                              ,pos_words_list,neg_words_list)#[0]
        bi_index = BI_func(stock_news_data_by_date)
        stock_news_data_by_date_with_BI.append([pd.to_datetime(dates), bi_index])
    stock_news_data_by_date_with_BI = pd.DataFrame(stock_news_data_by_date_with_BI, columns=["time_key", "bi_index"])

    return stock_news_data_by_date_with_BI

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1): # -1
		a = dataset[i:(i+look_back), ::]   # , [0,2,3,4,5,6]
		dataX.append(a)
		dataY.append(dataset[i + look_back, [1]])
	return np.array(dataX), np.array(dataY)

def create_Y_raw(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1): # -1
		dataY.append(dataset[i + look_back, [1]])  #[1] for index of close
	return np.array(dataY) # numpy.array(dataX),   

def get_stock_capital_flow(stock_name_id):#, start_date_analysis, end_date_analysis):#, start_date_ananlysis, end_date_ananlysis, delta_analysis_day):
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    ret, data = quote_ctx.get_capital_flow(stock_name_id,)
    if ret == RET_OK:
        print(data)
        print(data['in_flow'][0])    # 取第一条的净流入的资金额度
        print(data['in_flow'].values.tolist())   # 转为list
    else:
        print('error:', data)
    quote_ctx.close() # 结束后记得关闭当条连接，防止连接条数用尽
    return data

def get_all_stock_name(Market_type):
    # Market.HK Market.US SH沪股 SZ深股
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    ret, data = quote_ctx.get_stock_basicinfo(Market_type, SecurityType.STOCK)
    if ret == RET_OK:
        print(data)
    else:
        print('error:', data)
    print('******************************************')
    #ret, data = quote_ctx.get_stock_basicinfo(Market.HK, SecurityType.STOCK, ['HK.00647', 'HK.00700'])
    quote_ctx.close()
    return data

def get_stock_history_price(stock_name_id, start_date_analysis, end_date_analysis, delta_analysis_day):   # HK.00241
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    ret, data, page_req_key = quote_ctx.request_history_kline(stock_name_id, 
                                                              start=start_date_analysis, 
                                                              end=end_date_analysis, 
                                                              max_count=delta_analysis_day)  # 每页5个，请求第一页 
    # max_count=5: it will output a dataframe with each for 5 rows
    if ret == RET_OK:
        print('data:',data)
        print(data['code'][0])    # 取第一条的股票代码
        print(data['close'].values.tolist())   # 第一页收盘价转为list
    else:
        print('error:', data)
    while page_req_key != None:  # 请求后面的所有结果
        print('*************************************')
        ret, data, page_req_key = quote_ctx.request_history_kline(stock_name_id, 
                                                                  start=start_date_ananlysis, 
                                                                  end=end_date_ananlysis, 
                                                                  max_count=delta_analysis_day, 
                                                                  page_req_key=page_req_key) # 请求翻页后的数据
        if ret == RET_OK:
            print(data)
        else:
            print('error:', data)
    print('All pages are finished!')
    quote_ctx.close() # 结束后记得关闭当条连接，防止连接条数用尽
    # insert the holiday/ missing date and then replace value with the previous
    idx = pd.date_range(data['time_key'].min(), data['time_key'].max())
    data.index = pd.DatetimeIndex(data['time_key'])
    data = data.reindex(idx)
    data['time_key'] = data.index
    data = data.fillna(method='ffill')
    return data


current_path = os.getcwd()
#base_path = os.path.split(os.path.dirname(current_path))
base_path = os.path.dirname(current_path)
data_path = base_path + '\\data\\'
tool_path = base_path + '\\tool\\'
dict_path = base_path + '\\dict\\'
result_path = base_path + '\\result\\'
#pos_words = open(r"C:\Userhaoyu\working\investment\python script\Stock_Market_Sentiment_Analysis-master\dict\positive_extended.txt", "r", encoding='utf8').read()
#neg_words = open(r"C:\Userhaoyu\working\investment\python script\Stock_Market_Sentiment_Analysis-master\dict\negative_extended.txt", "r", encoding='utf8').read()
pos_words = open(dict_path + "positive_extended.txt", "r", encoding='utf8').read()
neg_words = open(dict_path + "negative_extended.txt", "r", encoding='utf8').read()

pos_words_list = re.split('\n| ',pos_words)
neg_words_list = re.split('\n| ',neg_words)

all_stock_name_HK = get_all_stock_name(Market.HK)
all_stock_name_SZ = get_all_stock_name(Market.SZ)  # SH # SZ
all_stock_name_SH = get_all_stock_name(Market.SH)  # SH # SZ

start_date_analysis = '2020-10-03'
end_date_analysis = '2020-12-24'
start_date_analysis_datetime = datetime.strptime(start_date_analysis, "%Y-%m-%d")
end_date_analysis_datetime = datetime.strptime(end_date_analysis, "%Y-%m-%d")
delta_analysis_day = end_date_analysis_datetime - start_date_analysis_datetime
delta_analysis_day = delta_analysis_day.days


news_data_path = data_path +'hang seng'

stock_news_data_filenames = os.listdir(news_data_path)
stock_news_data_filenames_list = [filename for filename in stock_news_data_filenames if filename.endswith('.csv')]
stock_news_data_name_list = [HanziConv.toSimplified(x.split('_google_')[0]) for x in stock_news_data_filenames_list]

combined_all_stock_history_date_news_biindex_list = []
for stock_news_file in stock_news_data_filenames_list:
    stock_news_data, stock_news_file_futu_stock_id = read_stock_news(news_data_path+r'\\'+stock_news_file)#阿里巴巴_google_news-2020-12-25-124230
    print('stock_news_file_futu_stock_id: ',stock_news_file_futu_stock_id)
    for i in range(0, len(stock_news_data)):
        date_str = stock_news_data[i]['time_key'].split("/")
        date_obj = date(int(date_str[2]), int(date_str[0]), int(date_str[1]))#+ timedelta(days=1) # 2020/12/23
        stock_news_data[i]['time_key'] = date_obj.strftime("%Y-%m-%d")#new_date_str
    date_tuple_list =  [x['time_key'] for x in stock_news_data]
    date_tuple_list = list(set(date_tuple_list))
    date_tuple_list = sorted(date_tuple_list) # ascending date
    stock_news_data_by_date_with_BI = calculate_BI_index(stock_news_data,date_tuple_list)
    stock_history_data_list = get_stock_history_price(stock_news_file_futu_stock_id, start_date_analysis, end_date_analysis, delta_analysis_day) #[dataframe]
    #stock_capital_flow_list = get_stock_capital_flow('HK.09988')#, start_date_analysis, end_date_analysis)
    #combined_stock_history_date_news_biindex = pd.merge(stock_history_data_list, stock_news_data_by_date_with_BI, right_index=True, left_index=True)
    combined_stock_history_date_news_biindex = pd.merge(stock_news_data_by_date_with_BI, stock_history_data_list, on=['time_key'])
    combined_all_stock_history_date_news_biindex_list.append((stock_news_file_futu_stock_id,combined_stock_history_date_news_biindex))



# fix random seed for reproducibility
np.random.seed(7)
look_back = 5
result = []
# load the dataset
# should add token distribution
#dataframe = read_csv(r'C:\Userhaoyu\working\investment\python script\Stock_Market_Sentiment_Analysis-master\merged_sentiment_idx.csv', engine='python') # C:\Userhaoyu\working\investment\python script\Stock_Market_Sentiment_Analysis-master
for combined_all_stock_history_date_news_biindex in combined_all_stock_history_date_news_biindex_list:
    dataframe = combined_all_stock_history_date_news_biindex[:][1]
    stock_code = combined_all_stock_history_date_news_biindex[:][0]
    dataframe_with_needed_col = dataframe[['bi_index', 'close', 'volume']] # ,'open','high','low','pe_ratio','turnover_rate'
    dataframe_with_needed_time = dataframe[['time_key']]
    dataset = dataframe_with_needed_col.values
    #dataset = dataset[0::,] # [1,3,4,5,6,7] # 1::
    #dataset = dataset[1::,:]
    # split into train and test sets
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    #train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    train, test_for_predict = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    #testX, testY_for_reality = create_dataset(test, look_back)
    
    scaler = MinMaxScaler() # feature_range=(1,3,4,5,6,7)
    dataset = scaler.fit_transform(dataset)  
    
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    
    
    testY_noscale = create_Y_raw(test_for_predict, look_back)
    testY_noscale = testY_noscale.astype('float32')
    testY_noscale_sacler = MinMaxScaler()
    #testY_scale_obj = testY_noscale_sacler.fit(X)
    test_for_predict2 = testY_noscale_sacler.fit_transform(testY_noscale)
    
    # split into train and test sets
    # reshape into X=t and Y=t+1
    look_back = 5
    #train, trainer_sacler = scaleing_data(train)
    #test, test_sacler = scaleing_data(test)
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    trainX = trainX.astype('float32')
    trainY = trainY.astype('float32')
    testX = testX.astype('float32')
    testY = testY.astype('float32')
    
    
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], look_back, trainX.shape[2]))  #5
    trainY = np.reshape(trainY, (trainY.shape[0], 1, trainY.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], look_back, testX.shape[2]))
    
    
    
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(5, input_shape=(look_back,trainX.shape[2]))) # 5, how many column for train x
    #model.add(LSTM(4, input_shape=(1, look_back)))
    
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict_sacled = model.predict(testX) # testX  # input test's X part to predict test'Y part
    # invert predictions
    # =============================================================================
    # trainPredict = scaler.inverse_transform(trainPredict)
    # trainY = scaler.inverse_transform([trainY])
    testPredicted = testY_noscale_sacler.inverse_transform(testPredict_sacled) # testY_noscale_sacler
    test_for_predict_actual_closed =  np.array([row[1] for row in test_for_predict])#test_for_predict[0:range(0,test_for_predict.shape[0])][1]
    print('testPredicted',testPredicted)
    print('test_for_predict_actual_closed',test_for_predict_actual_closed[4::])
    result.append((stock_code,testPredicted,test_for_predict_actual_closed))
# testPredict_sacled is better for getting the shape


"""biindex vs actual price"""
mpl.rcParams['axes.unicode_minus'] = False
#df = pd.read_csv('merged_sentiment_idx.csv', parse_dates=['created_time'])
#df.set_index(df.created_time, inplace=True)
#df = df.loc['2017-4-15':'2018-4-15']
for combined_data in combined_all_stock_history_date_news_biindex_list:
    #input()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    #scaler = MinMaxScaler()
    #scaler = preprocessing.MaxAbsScaler()
    stock_news_data_by_date_with_BI_biindex = combined_data[1]['bi_index']
    #stock_news_data_by_date_with_BI_biindex = scaler.fit_transform([stock_news_data_by_date_with_BI_biindex])  
    stock_news_data_by_date_with_BI_biindex_price = combined_data[1]['close']
    ax1.plot(range(0,stock_news_data_by_date_with_BI_biindex.shape[0]), stock_news_data_by_date_with_BI_biindex, color='#1F77B4', linestyle=':')
    ax2.plot(range(0,stock_news_data_by_date_with_BI_biindex_price.shape[0]), stock_news_data_by_date_with_BI_biindex_price, color='#4B73B1') # testY testY testY_noscale
    ax1.set_xlabel(r'Date')
    ax1.set_ylabel(r'BI_biindex')
    ax2.set_ylabel(r'Price')
    print(combined_data[1]['code'][0])
    ax1.title.set_text(str(combined_data[1]['code'][0]))
    plt.savefig(result_path+'png graph\\biindex v actual\\'+datetime.today().strftime("%Y-%m-%d")+'_'+combined_data[1]['code'][0]+'.png')
    plt.show()



"""lstm predicted vs actual price"""
mpl.rcParams['axes.unicode_minus'] = False
#df = pd.read_csv('merged_sentiment_idx.csv', parse_dates=['created_time'])
#df.set_index(df.created_time, inplace=True)
#df = df.loc['2017-4-15':'2018-4-15']
for predicted_result in result:
    #input()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    #scaler = MinMaxScaler()
    #scaler = preprocessing.MaxAbsScaler()
    #testPredicted,test_for_predict_actual_closed
    lstm_predicted = predicted_result[1]
    actual_price = predicted_result[2]
    for insert_lookback in range(0,look_back+2):  # insert days for prediction , added 7 day
        lstm_predicted = np.insert(lstm_predicted, 0, sum(lstm_predicted)/lstm_predicted.shape[0] )
    ax1.plot(range(0,lstm_predicted.shape[0]), lstm_predicted, color='#1F77B4', linestyle=':')
    ax2.plot(range(0,actual_price.shape[0]), actual_price, color='#4B73B1') # testY testY testY_noscale
    ax1.set_xlabel(r'Date')
    ax1.set_ylabel(r'Lstm Predicted')
    ax2.set_ylabel(r'Actual Price')
    print(predicted_result[0])
    ax1.title.set_text(str(predicted_result[0]))
    plt.savefig(result_path+'png graph\\lstm predicted vs actual price\\'+datetime.today().strftime("%Y-%m-%d")+'_'+predicted_result[0]+'.png')
    plt.show()









# =============================================================================
# 
# 
# 
# pos_words = open(r"C:\Userhaoyu\working\investment\python script\Stock_Market_Sentiment_Analysis-master\dict\positive_extended.txt", "r", encoding='utf8').read()
# neg_words = open(r"C:\Userhaoyu\working\investment\python script\Stock_Market_Sentiment_Analysis-master\dict\negative_extended.txt", "r", encoding='utf8').read()
# pos_words_list = re.split('\n| ',pos_words)
# neg_words_list = re.split('\n| ',neg_words)
# 
# # =============================================================================
# # start_date_analysis = '2020-10-03'
# # end_date_analysis = '2020-12-24'
# # =============================================================================
# start_date_analysis_datetime = datetime.strptime(start_date_analysis, "%Y-%m-%d")
# end_date_analysis_datetime = datetime.strptime(end_date_analysis, "%Y-%m-%d")
# delta_analysis_day = end_date_analysis_datetime - start_date_analysis_datetime
# delta_analysis_day = delta_analysis_day.days
# 
# 
# =============================================================================

# algorithm notes
# get the trend is better for not scale back
# track the big capital flow in and out(seperately)
# six to six predict for two weeks



