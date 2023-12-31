'''
LIBRARY
'''
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
# visualisation
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go



'''
PARAMETERS
'''
# Tickers
tickers = ['MSFT','^SPX','^SPY']
start = '2012-12-20'
end = '2022-12-20'

# msft = yf.download('MSFT')
# msft_info = yf.Ticker('MSFT').info
# msft.head()



'''
DATA
'''
def get_df(tkr):
    df = yf.download(tkr,start,end)
    df['Return'] = df['Close'].pct_change()
    df['Signed_Vol'] = np.sign(df['Return'])*df['Volume']
    df['Vol'] = df['Close'].rolling(5).std()
    return df

spx = get_df('^SPX')

def strategy(df):
    df['SMA5_Price'] = df['Close'].rolling(5).mean()
    df['SMA5_Vol'] = df['Vol'].rolling(5).mean()
    df['SMA5_Volume'] = df['Signed_Vol'].rolling(5).mean()

    df['Signal'] = 0

    # buy signal
    df.loc[(df['SMA5_Price']<df['Close']) & (df['SMA5_Volume']>0),'Signal']=1

    # sell signal
    df.loc[(df['SMA5_Price']>df['Close']) & (df['SMA5_Vol']<df['Vol']) & (df['SMA5_Volume']<0),'Signal']=-1

    df['Trade'] = df['Signal']*df['Close']

    return df

spx_strat = strategy(spx)
# spx_strat = spx_strat[spx_strat['Signal']!=0]


'''
VISUALIZATION
'''
col = 'Close'
df = spx_strat
df.index = pd.to_datetime(df.index)
df = df.reset_index()

fig = make_subplots(rows=4, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05)

## Subplot-1
### Price series
fig.add_trace(go.Scatter(x=df['Date'], y=(df[col]),
                         line_color='green',
                         mode='lines',
                         showlegend=True,
                         name="close"
                         ),
              row=1, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=(df[col].rolling(5).mean()),
                         line_color = 'blue',
                         mode = 'lines',
                         showlegend = True,
                         name = "5D"
                        ),
              row=1,col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=(df[col].rolling(25).mean()),
                         line_color = 'red',
                         mode = 'lines',
                         showlegend = True,
                         name = "25D"
                        ),
              row=1,col=1)

## Subplot-2
### Volume series
fig.add_trace(go.Scatter(x=df['Date'], y=(df['Volume']),
                         line_color = 'green',
                         mode = 'lines',
                         showlegend = True,
                         name = "Volume"
                        ),
              row=2,col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=(df['Signed_Vol'].rolling(5).mean()),
                         line_color = 'blue',
                         mode = 'lines',
                         showlegend = True,
                         name = "5D"),
              row=2,col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=(df['Signed_Vol'].rolling(25).mean()),
                         line_color = 'red',
                         mode = 'lines',
                         showlegend = True,
                         name = "25D"),
              row=2,col=1)


## Subplot-3
### Volatility series
fig.add_trace(go.Scatter(x=df['Date'], y=(df['Vol']),
                         line_color = 'green',
                         mode = 'lines',
                         showlegend = True,
                         name = "Vol5",
                         ),
              row=3,col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=(df['Vol'].rolling(5).mean()),
                         line_color = 'blue',
                         mode = 'lines',
                         showlegend = True,
                         name = "5DVol5",
                         ),
              row=3,col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=(df['Vol'].rolling(25).mean()),
                         line_color = 'red',
                         mode = 'lines',
                         showlegend = True,
                         name = "25DVol5",
                         ),
              row=3,col=1)


fig.add_trace(go.Scatter(x=df['Date'], y=(df['Signal']),
                         line_color = 'red',
                         mode = 'lines',
                         showlegend = True,
                         name = "25DVol5",
                         ),
              row=4,col=1)


fig.update_traces(xaxis='x1')
fig.update_layout(height=800,
                  hovermode='x unified',
                  hoverlabel=dict(bgcolor='rgba(255,255,255,0.75)',
                                  font=dict(color='black'))
                  )
date_format = "%b %d, %Y"
fig.update_xaxes(tickformat=date_format,showticklabels=True)

fig.show()

import plotly.io as pio
pio.renderers.default = "browser"

# spx.to_csv('trades_nify.csv')


'''
IF WANT PLOTS WITH MULTIPLE X AXIS
'''
# fig = make_subplots(rows=2, cols=1,
#                     shared_xaxes=True,
#                     vertical_spacing=0.05)
#
# fig['layout']['xaxis1'].update(title='Date')
# fig['layout']['xaxis2'].update(title='Date')
# fig['layout']['yaxis1'].update(title='Price')
# fig['layout']['yaxis2'].update(title='Volume')
#
# for k in range(1,2):
#     fig['layout'].update({'yaxis{}'.format(k+2): dict(anchor='x'+str(k),
#                                                           overlaying='y'+str(k),
#                                                           side='right',
#                                                          )
#                             })
# fig['layout']['yaxis3'].update(title='Vol',range=[0,200])
#
# # fig['layout'].update({'yaxis3'.format(k+4): dict(anchor='x'+str(k),
# #                                                           overlaying='y'+str(k),
# #                                                           side='right',
# #                                                          )
# #                             })
#
# # layout = go.Layout(title='CLE vs Model',
# #                    yaxis=dict(title='Crude and Model'),
# #                    yaxis2=dict(title='Moddel Difference',
# #                                overlaying='y',
# #                                side='right'))
#
# ## Subplot-1
# ### close series
# fig.add_trace(go.Scatter(x=df['Date'], y=(df[col]),
#                          line_color='green',
#                          mode='lines',
#                          showlegend=True,
#                          name="close"
#                          ),
#               row=1, col=1)
# fig.add_trace(go.Scatter(x=df['Date'], y=(df[col].rolling(5).mean()),
#                          line_color = 'blue',
#                          mode = 'lines',
#                          showlegend = True,
#                          name = "5D"
#                         ),
#               row=1,col=1)
# fig.add_trace(go.Scatter(x=df['Date'], y=(df[col].rolling(25).mean()),
#                          line_color = 'red',
#                          mode = 'lines',
#                          showlegend = True,
#                          name = "25D"
#                         ),
#               row=1,col=1)
# fig.add_trace(go.Scatter(x=df['Date'], y=(df[col].rolling(5).std()),
#                          line_color = 'pink',
#                          mode = 'lines',
#                          showlegend = True,
#                          name = "5DV",
#                         yaxis='y3'
#                          ),
#
#               row=1,col=1)
#
# ## subplot-2
# ### volume series
# fig.add_trace(go.Scatter(x=df['Date'], y=(df['Volume']),
#                          line_color = 'green',
#                          mode = 'lines',
#                          showlegend = True,
#                          name = "Volume",
#                          yaxis='y2'
#                         ),
#               row=2,col=1)
# fig.add_trace(go.Scatter(x=df['Date'], y=(df['Signed_Vol'].rolling(5).mean()),
#                          line_color = 'blue',
#                          mode = 'lines',
#                          showlegend = True,
#                          name = "5D",
#                          yaxis='y2'),
#               row=2,col=1)
# fig.add_trace(go.Scatter(x=df['Date'], y=(df['Signed_Vol'].rolling(25).mean()),
#                          line_color = 'red',
#                          mode = 'lines',
#                          showlegend = True,
#                          name = "25D",
#                          yaxis='y2'),
#               row=2,col=1)
#
# fig['data'][1].update(yaxis='y'+str(3))
# fig.update_traces(xaxis='x1')
# fig.update_layout(height=800,
#                   hovermode='x unified')
# date_format = "%b %d, %Y"
# fig.update_xaxes(tickformat=date_format,showticklabels=True)
#
# fig.show()
#
# import plotly.io as pio
# pio.renderers.default = "browser"
