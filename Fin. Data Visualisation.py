import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import matplotlib.dates as dt
import talib
import mplfinance as fplt
from mpl_toolkits import mplot3d
from datetime import datetime
from itertools import chain
from matplotlib import cm

# Q1.2 Yahoo Finance
# Obtaining Price Information History
ticker_name1 = 'GS'
yticker1 = yf.Ticker(ticker_name1)
GS = yticker1.history(period="1y")

ticker_name2 = 'SPY'
yticker2 = yf.Ticker(ticker_name2)
SPY = yticker2.history(period="1y")

# Calculating Log Returns from Price Information History
GS['Return'] = np.log(GS['Close']/GS['Close'].shift(1))
R_GS = GS['Return']
Y = R_GS[1:]

SPY['Return'] = np.log(SPY['Close']/SPY['Close'].shift(1))
R_SPY= SPY['Return']
X = R_SPY[1:]

# Converting to LaTex table
LR_SPY = X.to_frame()
print(LR_SPY.to_latex())

# Converting to LaTex table
LR_GS = Y.to_frame()
print(LR_GS.style.to_latex())

# Q1.3 Scatter Plot - GS v SPY500
fig1, ax1 = plt.subplots()
a, b = np.polyfit(X, Y, 1)
plt.scatter(X, Y, label=f'Correlation = {np.round(np.corrcoef(X,Y)[0,1], 2)}')
plt.plot(X, a*X + b, color='orange', linestyle='--', linewidth=2)
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':150})
plt.title('The Goldmansachs Group vs SPY Returns (Mar 2022 - Mar 2023)')
plt.xlabel('SPY Returns')
plt.ylabel('Goldmansachs Returns')
plt.legend(loc='upper left')
plt.show()

# Q1.4 Histogram - Compare GS Returns to S&P500 Returns
# fig2, ax2 = plt.subplots(1, 2, sharey=True)
# ax2[0].hist(Y, bins=20)
# ax2[1].hist(X, bins=20)
# ax2[0].set(xlabel="GS")
# ax2[1].set(xlabel="SPY")
# ax2[0].title.set_text("Goldmansachs Returns (Feb 2022 - Feb 2023)")
# ax2[1].title.set_text("S&P500 Returns (Feb 2022 - Feb 2023)")
# plt.title('Goldmansachs and S&P500 Returns 28/02/2022 - 28/02/2023')

# Q1.4 Histogram - GS Returns
fig2, ax2 = plt.subplots()
ax2.hist(Y, bins=20)
ax2.set(xlabel = "GS Returns")
ax2.set(ylabel = "Frequency")
plt.title('Goldmansachs Returns 01/03/2022 - 01/03/2023')
plt.show()

# Q1.5 Box Plots
fig3, ax3 = plt.subplots(1, 2, sharey=True)
ax3[0].boxplot(X)
ax3[1].boxplot(Y)
ax3[0].set(xlabel="SPY")
ax3[1].set(xlabel="GS")
ax3[0].title.set_text("S&P500 Returns (Mar 2022 - Mar 2023)")
ax3[1].title.set_text("Goldmansachs Returns (Mar 2022 - Mar 2023)")
plt.show()

# Q1.6 Generating Buy/Sell Signals - Moving Average Indicator
# Stock Price Data
df = pd.DataFrame(GS)

# Store Simple Moving Average (SMA) Data
df['SMA20'] = talib.SMA(GS.Close, 20)
df['SMA50'] = talib.SMA(GS.Close, 50)

# Obtain Buy and Sell Signals
df['Signal'] = np.where(df['SMA20'] > df['SMA50'], 1, 0)
df['Position'] = df['Signal'].diff()
df['Buy'] = np.where(df['Position'] == 1, df['Close'], np.NAN)
df['Sell'] = np.where(df['Position'] == -1, df['Close'], np.NAN)

# Plot Stock Price, SMA and Signals
df = pd.DataFrame(GS)
plt.figure(figsize=(16,8))
plt.title('Goldmansachs - Stock Price History with Buy & Sell Signals', fontsize=18)
plt.plot(df['Close'], alpha = 0.7, label='Close')
plt.plot(df['SMA20'], alpha = 0.7, label='SMA20')
plt.plot(df['SMA50'], alpha = 0.7, label='SMA50')
plt.scatter(df.index, df['Buy'], alpha = 1, label = 'Buy Signal', marker = '^', color = 'green')
plt.scatter(df.index, df['Sell'], alpha = 1, label = 'Sell Signal', marker = 'v', color = 'red')
plt.xlabel('Date', fontsize = 16)
plt.ylabel('Close Price', fontsize = 16)
plt.legend()
plt.show()

# Q1.7 3D Implied Volatility Surface
# Extraction of Option Data
sTicker = "GS"
stock = yf.Ticker(sTicker)

# Store the Maturities of Options
lMaturity = list(stock.options)
today = datetime.now().date()

lDTE = [] # DTE = Days to Expiration
lData_CALL = []

for maturity in lMaturity:
    maturity_date = datetime.strptime(maturity, '%Y-%m-%d').date()
    lDTE.append((maturity_date - today).days)
    # Store Data for Calls
    lData_CALL.append(stock.option_chain(maturity).calls)

# View all Option Data
print(lData_CALL)

# Store Data in Lists
# Empty Lists
lStrike = []
lDTE_ext = []
lImpVol = []

for i in range(0,len(lData_CALL)):
    lStrike.append(lData_CALL[i]["strike"])
    lDTE_ext.append(np.repeat(lDTE[i], len(lData_CALL[i])))
    lImpVol.append(lData_CALL[i]["impliedVolatility"])

# Delist 'List of Lists'
lStrike = list(chain(*lStrike))
lDTE_ext = list(chain(*lDTE_ext))
lImpVol = list(chain(*lImpVol))

# Prepare Plot Title
S = GS['Close'].values[-1]
T = today

# Plot 3-D Imp Vol Surface
fig5 = plt.figure(figsize=(15,15))
axs = fig5.add_subplot(111, projection='3d')
axs.plot_trisurf(lStrike, lDTE_ext, lImpVol, cmap=cm.jet)
axs.view_init(30, 65)
axs.set_xlabel('Strike')
axs.set_ylabel('Days to Expiration')
axs.set_zlabel('Implied Volatility for Call Options')
axs.set_title("Implied Volatility Surface for "+str(sTicker)+"\n Current Price: $"+str(round(S, ndigits=4))+" as of Date: "+str(T)+"", fontsize=15)
# plt.title('Implied Volatility Surface for %s - Current Price: %s - Date: %s' %
#          (sTicker, '{0:.4g}'.format(stock['Close'].values[-1]), today))
plt.show()