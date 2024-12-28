import pandas as pd
data = pd.read_csv('/Users/yaotian/Desktop/Transactions.csv')
print(data.head())
print(data.info())
print(data.isnull().sum())
data['instance_date'] = pd.to_datetime(data['instance_date'], errors='coerce', dayfirst=True)
data = data.dropna(subset=['instance_date'])
data['month'] = data['instance_date'].dt.to_period('M')
monthly_trends = data.groupby('month').size()
print(monthly_trends)

import matplotlib.pyplot as plt
plt.figure(1)
monthly_trends.plot(kind='line', figsize=(12,6))
plt.title('Monthly Transaction Trends')
plt.xlabel('Month')
plt.ylabel('Number of Transactions')
plt.grid()

data['month_only'] = data['instance_date'].dt.month
seasonal_trends = data.groupby('month_only').size()
import matplotlib.pyplot as plt
plt.figure(2)
seasonal_trends.plot(kind='bar', figsize=(10,6))
plt.title('Seasonal Trends: average Monthly Transaction Volume')
plt.xlabel('Month')
plt.ylabel('Number of Transactions')
plt.xticks(range(0,12),['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],rotation=45)
plt.grid(axis='y')


from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
forecast_data = monthly_trends.reset_index()
forecast_data['month'] = forecast_data['month'].dt.to_timestamp()
forecast_data.columns = ['ds','y']
model = Prophet(yearly_seasonality=True, daily_seasonality=False)
model.fit(forecast_data)
future = model.make_future_dataframe(periods=12,freq='ME')
forecast = model.predict(future)

fig = model.plot(forecast, figsize=(12,6))
ax = fig.gca()
ax.set_title('Transaction Volume Prediction', fontsize=16)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Number of Transactions', fontsize=12)
ax.grid()
plt.show()