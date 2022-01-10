from numpy.lib.function_base import select
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import pickle    
import warnings
warnings.filterwarnings('ignore')


class model:
    def __init__(self):
        self.dicts = {}
        self.sales = 0

    def stat_test(self ,sales_data):

        
        print("Augmented Dickey-fuller test result: ")
        result = adfuller(sales_data, autolag="AIC")

        print("ADF test statistic: ", result[0])
        print("p-value:", result[1])

        print("Critical Values:")
        for key, val in result[4].items():
            print("\t%s : %f" %(key, val))
            self.dicts[key] = val

        Dicky_fuller_test = { "ADF test statistic" : result[0], "p-value": result[1], "Critical Values": self.dicts}
        return Dicky_fuller_test

    def train(self ,df):
        df['order_date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
        df['ship_date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)
        df['YearMonth'] = df['order_date'].apply(lambda x: x.strftime("%Y-%m"))
        features = ['Order ID','Customer ID', 'Product ID', 'order_date', 'ship_date', 'Product Name', 'Country', 'Region', 'State', 'City', 
            'Segment', 'Category', 'Sub-Category', 'Ship Mode', 'YearMonth', 'Order Year', 'Order Month', 'Sales']
        df1= df[features]
        if df1.duplicated().sum() > 0:
            df1 = df1.drop_duplicates()
        # prepare data
        sales_data = df1[['order_date', 'Sales']]
        sales_data = sales_data.set_index('order_date')

        # calculating rolling statistics.
        roll_mean = sales_data.rolling(window=7).mean()
        roll_std = sales_data.rolling(window=7).std()

        # plotting rolling statistics with orignal data mean.
        plt.figure(figsize=(14, 7), dpi=100)
        data_mean = plt.plot(sales_data.resample('W').mean(), label='Original', marker="o", alpha=0.5)
        mean = plt.plot(roll_mean.resample('W').mean(), label="Rolling Mean", marker=".")
        std = plt.plot(roll_std.resample('W').std(), label="Rolling Standard", alpha=0.5)

        plt.title("Rolling Mean Test")
        plt.legend()
        plt.savefig('static/img/rolling_stats.png')

        Dicky_fuller_test = self.stat_test(sales_data)
        sales = pd.DataFrame(df1.groupby(by=['order_date']).sum()['Sales'])
        model = sm.tsa.statespace.SARIMAX(sales,order=(1, 0, 0), seasonal_order=(1, 1, 1, 12))
        result = model.fit()
        # Visualization of the performance of our model
        result.plot_diagnostics(figsize=(14, 10))
        plt.savefig('static/img/performance.png')

        with open('learned_model.pkl','wb') as f:
            pickle.dump(result, f)

        sales.to_csv('sales.csv')
        print("Model is trained")

        return Dicky_fuller_test
        






    def predict(self, lr, dataset, start_date, end_date):
        # load saved model
        l = list(lr.predict(start=start_date, end=end_date, dynamic=False))

        index1 = dataset['order_date'][dataset['order_date']==start_date].index
        index2 = dataset['order_date'][dataset['order_date']==end_date].index

        actual = dataset.loc[index1.tolist()[0]:index2.tolist()[0]]['Sales']
        data = pd.DataFrame(actual)
        data['Forecast'] = l
        
        #visualization for the same
        data.plot(figsize=(14, 8))

        data.to_csv('results.csv')
        plt.savefig('static/img/sales_prediction.png')
       
        actual = data['Sales']
        preds = data['Forecast']
        
        rmse_sarima = np.sqrt(mean_squared_error(preds, actual))
        print("Root Mean Squared Error for SARIMAX:", rmse_sarima)

        return  rmse_sarima
        

