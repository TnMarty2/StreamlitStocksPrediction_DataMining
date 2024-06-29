# pip install streamlit fbprophet yfinance plotly statsmodels
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import statsmodels.api as sm
import pandas as pd

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('BBCA.JK', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Slider for moving average window size
ma_window = st.slider('Moving Average Window Size:', 1, 100, 20)

# Calculate moving average
data['Moving_Avg'] = data['Close'].rolling(window=ma_window).mean()

# Plot raw data with moving average
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Moving_Avg'], name="moving_avg", line=dict(color='orange')))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Select forecasting method
method = st.selectbox('Select forecasting method', ('Prophet', 'ARIMA'))

# Prepare data for forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

if method == 'Prophet':
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    
    # Show and plot forecast
    st.subheader('Forecast data (Prophet)')
    st.write(forecast.tail())
    
    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    
    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

elif method == 'ARIMA':
    # Fit ARIMA model
    df_train.set_index('ds', inplace=True)
    model = sm.tsa.ARIMA(df_train['y'], order=(5, 1, 0))
    model_fit = model.fit()
    
    # Make forecast
    forecast = model_fit.forecast(steps=period)
    forecast_dates = pd.date_range(start=df_train.index[-1], periods=period+1, inclusive='right')
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})
    
    # Show and plot forecast
    st.subheader('Forecast data (ARIMA)')
    st.write(forecast_df.tail())
    
    st.write(f'Forecast plot for {n_years} years')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_train.index, y=df_train['y'], name='Observed'))
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], name='Forecast', line=dict(color='red')))
    fig.layout.update(title_text='ARIMA Forecast', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
