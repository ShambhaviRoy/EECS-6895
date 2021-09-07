#Two Commands to execute:
#pip install streamlit
#streamlit run front_end.py
import numpy as np
import matplotlib.pyplot as plt 
import streamlit as st
import os
import pandas as pd
from PIL import Image
import tensorflow as tf
import datetime as dt
import pandas_datareader.data as web

from datetime import date
from datetime import datetime
from datetime import timedelta

from sklearn.preprocessing import StandardScaler

#Five different LSTM models
LSTM_close = tf.keras.models.load_model("./models/close_model_2")
LSTM_open = tf.keras.models.load_model("./models/open_model_2")
LSTM_high = tf.keras.models.load_model("./models/high_model_2")
LSTM_low = tf.keras.models.load_model("./models/low_model_2")
LSTM_volume = tf.keras.models.load_model("./models/volume_model_2")

#List of S&P 500 Tickers

list_of_tickers = ['MMM', 'ABT', 'ABBV', 'ABMD', 'ACN', 'ATVI', 'ADBE', 'AMD', 'AAP', 'AES', 'AFL', 'A', 'APD', 'AKAM', 'ALK', 'ALB', 'ARE', 'ALXN', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'ANTM', 'AON', 'AOS', 'APA', 'AAPL', 'AMAT', 'APTV', 'ADM', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'BKR', 'BLL', 'BAC', 'BK', 'BAX', 'BDX', 'BRK.B', 'BBY', 'BIO', 'BIIB', 'BLK', 'BA', 'BKNG', 'BWA', 'BXP', 'BSX', 'BMY', 'AVGO', 'BR', 'BF.B', 'CHRW', 'COG', 'CDNS', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CTLT', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 'CNC', 'CNP', 'CERN', 'CF', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CTXS', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'COP', 'ED', 'STZ', 'COO', 'CPRT', 'GLW', 'CTVA', 'COST', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DAL', 'XRAY', 'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DISCA', 'DISCK', 'DISH', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DTE', 'DUK', 'DRE', 'DD', 'DXC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMR', 'ENPH', 'ETR', 'EOG', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ETSY', 'EVRG', 'ES', 'RE', 'EXC', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FB', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FE', 'FRC', 'FISV', 'FLT', 'FLIR', 'FLS', 'FMC', 'F', 'FTNT', 'FTV', 'FBHS', 'FOXA', 'FOX', 'BEN', 'FCX', 'GPS', 'GRMN', 'IT', 'GD', 'GE', 'GIS', 'GM', 'GPC', 'GILD', 'GL', 'GPN', 'GS', 'GWW', 'HAL', 'HBI', 'HIG', 'HAS', 'HCA', 'PEAK', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HFC', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUM', 'HBAN', 'HII', 'IEX', 'IDXX', 'INFO', 'ITW', 'ILMN', 'INCY', 'IR', 'INTC', 'ICE', 'IBM', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IPGP', 'IQV', 'IRM', 'JKHY', 'J', 'JBHT', 'SJM', 'JNJ', 'JCI', 'JPM', 'JNPR', 'KSU', 'K', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KLAC', 'KHC', 'KR', 'LB', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LEG', 'LDOS', 'LEN', 'LLY', 'LNC', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LUMN', 'LYB', 'MTB', 'MRO', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MKC', 'MXIM', 'MCD', 'MCK', 'MDT', 'MRK', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MHK', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NWL', 'NEM', 'NWSA', 'NWS', 'NEE', 'NLSN', 'NKE', 'NI', 'NSC', 'NTRS', 'NOC', 'NLOK', 'NCLH', 'NOV', 'NRG', 'NUE', 'NVDA', 'NVR', 'ORLY', 'OXY', 'ODFL', 'OMC', 'OKE', 'ORCL', 'OTIS', 'PCAR', 'PKG', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PBCT', 'PEP', 'PKI', 'PRGO', 'PFE', 'PM', 'PSX', 'PNW', 'PXD', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RHI', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SEE', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SLG', 'SNA', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STE', 'SYK', 'SIVB', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TGT', 'TEL', 'TDY', 'TFX', 'TER', 'TSLA', 'TXN', 'TXT', 'TMO', 'TJX', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TWTR', 'TYL', 'TSN', 'UDR', 'ULTA', 'USB', 'UAA', 'UA', 'UNP', 'UAL', 'UNH', 'UPS', 'URI', 'UHS', 'UNM', 'VLO', 'VAR', 'VTR', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VFC', 'VIAC', 'VTRS', 'V', 'VNT', 'VNO', 'VMC', 'WRB', 'WAB', 'WMT', 'WBA', 'DIS', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WU', 'WRK', 'WY', 'WHR', 'WMB', 'WLTW', 'WYNN', 'XEL', 'XRX', 'XLNX', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZION', 'ZTS']
list_of_tickers.remove("BRK.B")
list_of_tickers.remove("BF.B")

#Add title and image
st.write("""
# EECS 6895 AI Trader Web Application
## Authored by Richard Samoilenko, Shambhavi Roy, and Meet Desai
Date range from Jan 3, 2000 - Present \n
**Trading equities, mutual funds, index and exchange traded funds, cryptocurrency, foreign exchange, options, binary options or futures, on margin carries a high level of risk, and may not be suitable for all investors. The high degree of leverage can work against you as well as for you. Before deciding to invest you should carefully consider your investment objectives, level of experience, and risk appetite. The possibility exists that you could sustain a loss of some or all of your initial investment and therefore you should not invest money that you cannot afford to lose. You should be aware of all the risks associated with trading and seek advice from an independent financial advisor if you have any doubts. No representation is being made that any account will or is likely to achieve profits or losses similar to those discussed on this website. The past performance of any trading system or methodology is not necessarily indicative of future results.**
""")

#Create sidebar header
st.sidebar.header("User Input")

#Create a function to get the users input
def get_input():
    #Temporary code for getting the date today
    today = date.today()
    today = today.strftime("%Y-%m-%d")
    
    start_date = st.sidebar.text_input("Start Date", "2000-01-03")
    end_date = st.sidebar.text_input("End Date", today)
    increment = st.sidebar.selectbox("Data Increment", ("Daily","Hourly"))
    stock_symbol = st.sidebar.selectbox("Stock Symbol", list_of_tickers)

    return start_date, end_date, stock_symbol, increment

#Ask user for end date of forecasting and 
def get_forecasting_input():
    today = date.today()
    today = today.strftime("%Y-%m-%d")
    forecasting_end = st.sidebar.text_input("Forecasting End Date", (date.today() + timedelta(days=25)).strftime("%Y-%m-%d"))
    mode = st.sidebar.selectbox("Forecasting Mode", ("High","Low", "Open", "Volume", "Close"))
    
    current_date = datetime.strptime(today, "%Y-%m-%d")
    end_date = datetime.strptime(forecasting_end, "%Y-%m-%d")
    if current_date.weekday() == 5:
        current_date = current_date - timedelta(days=1)
    elif current_date.weekday() == 6:
        current_date = current_date - timedelta(days=2)
    
    steps = 0
    while current_date != end_date:
        current_date = current_date + timedelta(days = 1)
        if current_date.weekday() == 5:
            current_date = current_date + timedelta(days = 2)
        elif current_date.weekday() == 6:
            current_date = current_date + timedelta(days = 1)
        steps += 1
        if current_date >= end_date:
            break
    
    print("Steps: ", steps)
    return forecasting_end, steps, mode

#Function for company name
#TODO: Create dictionary for retrieving company names
def get_company_name(symbol):
    if symbol == "AAPL":
        return 'Apple'
    elif symbol == "GOOG":
        return 'Google'
    else:
        return symbol

#Function to get proper data and timefrom from user start data and end date
def get_data(symbol, start, end):
    #Load Data
    
    #TODO: Add a line of code to take the DataFrame of the symbol stock only
    
    if symbol.upper() in list_of_tickers:
        df = web.DataReader(symbol, 'yahoo', start, end)
        df.reset_index(level=0, inplace=True)
        #print(df.head())
        #df = pd.read_csv("./data_generation/Stock_data/Daily_data_" + symbol.upper() + ".csv")
    else:
        df = pd.DataFrame(columns = ['Date', 'Close', "Open", "Volume", "High", "Low"])
        
    #Get data range
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    
    #Set start and end index rows both to 0
    start_row = 0; end_row = 0
    
    #Start date from top of the dataset and go down
    #Find start date
    for i in range(len(df)):
        if start <= pd.to_datetime(df['Date'][i]):
            start_row = i; break;
            
    #Find end date
    for i in range(len(df)):
        if end == pd.to_datetime(df['Date'][i]):
            end_row = i; break;
            
    #Set index to be the date
    df = df.set_index(pd.DatetimeIndex(df["Date"].values))
    
    return df.iloc[start_row:end_row + 1,:]

#TODO: Use model to make predictions
def make_prediction(symbol,forecasting_end, steps, df, mode="Close"):
    
    df = df[["High", "Low", "Open", "Volume", "Close"]].reset_index(drop=True)

    scaler = StandardScaler().fit(df)
    new_df = scaler.transform(df)
    
    print(new_df)
    
    new_df = new_df[-7:, 0:5]
    
    current_steps = 0
    
    while current_steps < steps:
        
        #window = new_df[-7:,0:5].reshape((1,7,5))
        #predicted = LSTM(window)[0]
        #new_df = np.vstack([new_df, predicted])
        
        window = new_df[-7:,0:5]
        
        
        #window_high = window[:,[1,2,3,4]].reshape((1,7,4))
        #window_low =  window[:,[0,2,3,4]].reshape((1,7,4))
        #window_open =  window[:,[0,1,3,4]].reshape((1,7,4))
        #window_volume =  window[:,[0,1,2,4]].reshape((1,7,4))
        #window_close =  window[:,[0,1,2,3]].reshape((1,7,4))
        
        
        window_high = window[:,0:5].reshape((1,7,5))
        window_low =  window[:,0:5].reshape((1,7,5))
        window_open =  window[:,0:5].reshape((1,7,5))
        window_volume =  window[:,0:5].reshape((1,7,5))
        window_close =  window[:,0:5].reshape((1,7,5))
        
        predicted_high = LSTM_high(window_high)[0][0]
        predicted_low = LSTM_low(window_low)[0][0]
        predicted_open = LSTM_open(window_open)[0][0]
        predicted_volume = LSTM_volume(window_volume)[0][0]
        predicted_close = LSTM_close(window_close)[0][0]

        add_array = np.array([predicted_high, predicted_low, predicted_open, predicted_volume, predicted_close])
        
        new_df = np.vstack([new_df, add_array])
        
        current_steps += 1
    
    new_df = new_df.reshape(-1,5)
    print("Before inverse transform: ", new_df)
    new_df = scaler.inverse_transform(new_df)
    
    data = {"High" : new_df[:,0], "Low" : new_df[:,1],"Open" : new_df[:,2],"Volume" : new_df[:,3],"Close" : new_df[:,4],}
    new_df = pd.DataFrame(data)
    print("Predictions: ", new_df)
    
    return new_df

def buy_sell_signal_plot(forecasted_df, mode, score):
    
    window_s = 0; window_l = 0;
    if score < 2:
        window_s = 10; window_l = 20;
    elif score < 7: 
        window_s = 3; window_l = 5
    #elif score < 15:
    #    window_s = 6; window_l = 13
    else: 
        window_s = 2; window_l = 3
    
    
    #Short Term Simple Average 
    forecasted_df["10_Day_SMA"] = forecasted_df["Close"].rolling(window = window_s, min_periods=1).mean()
    #Long Term Simple Average
    forecasted_df["30_Day_SMA"] = forecasted_df["Close"].rolling(window = window_l, min_periods=1).mean()
    
    #Create Position from Signal; +1 is buy, -1 is sell
    forecasted_df["Signal"] = np.where(forecasted_df["10_Day_SMA"] > forecasted_df["30_Day_SMA"], 1, 0)
    forecasted_df["Position"] = forecasted_df["Signal"].diff()
    
    #Plot predicted time series
    plt.plot(np.arange(0,forecasted_df.shape[0]), forecasted_df[mode])
    #Plot buy/sell markers
    plt.plot(forecasted_df[forecasted_df["Position"] == 1].index,forecasted_df['10_Day_SMA'][forecasted_df['Position'] == 1], 
         '^', markersize = 15, color = 'g', label = 'buy')
    plt.plot(forecasted_df[forecasted_df["Position"] == -1].index,forecasted_df['10_Day_SMA'][forecasted_df['Position'] == -1], 
         'v', markersize = 15, color = 'r', label = 'sell')


#Get user input
start, end, symbol, increment = get_input()
#Get Forecasting End
forecasting_end, steps, mode = get_forecasting_input()
#Get date
df = get_data(symbol, start, end)
#print(df.head(5))
#Get the company name
company_name = get_company_name(symbol.upper())

#Display Close Price
st.header(company_name + " " + mode + " Price\n")
st.line_chart(df[mode])

#User Survey
st.title("User Risk Survey\n")
st.header("Please select the responses that best match your own preferences\n")

responses = []
responses_dict = {0 : ["$0-$1000", "$1000-$10000", "$10000-$25000", "$25000+"], 1: ["Need money at all times", "0-5 years", "5-10 years", "10+ years"], 2 : ['1','2','3','4','5'], 3: ["Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"], 4: ["Transfer your money to a more secure investment", "Hold your investment and look for improvement", "Invest more to take advantage of lower price"] }

total_capital_response = st.select_slider("Which of these ranges best describes your current total capital?", options=["$0-$1000", "$1000-$10000", "$10000-$25000", "$25000+"]); responses.append(total_capital_response);
length_of_investment_response = st.select_slider("How long are you looking to invest your money for?", ["Need money at all times", "0-5 years", "5-10 years", "10+ years"]); responses.append(length_of_investment_response);
day_trading_response = st.select_slider('How likely are you to trade more than five times in a given day?', options=['1','2','3','4','5']);
responses.append(day_trading_response);
certainty_response = st.selectbox("I would prefer small certain gains to large uncertain ones.", ["Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"]); responses.append(certainty_response);
future_investment_response = st.selectbox("Imagine that three months after making an investment the financial markets start to perform badly, causing your own investment to go down by a significant amount. What would your reaction be?", ["Transfer your money to a more secure investment", "Hold your investment and look for improvement", "Invest more to take advantage of lower price"]); responses.append(future_investment_response);

#Higher risk score means more risky
risk_score = 0

for i in range(len(responses)):
    score = responses_dict[i].index(responses[i])
    risk_score += score
    
print("The risk score is: ", risk_score)

#Code for displaying forecasting
st.title("Time Series Forecasting")
st.header("Price Prediction for " + company_name + " " + mode + " Price")

st.subheader("Forecasting begins at " + date.today().strftime("%Y-%m-%d") + " and ends at " + forecasting_end)

#Get forecasted dataframe
st.set_option('deprecation.showPyplotGlobalUse', False)
forecasted_df = make_prediction(symbol, forecasting_end, steps, df)
#forecasted_df.plot(kind="line", y = mode)
#st.pyplot()

buy_sell_signal_plot(forecasted_df, mode, risk_score)
st.pyplot()

    