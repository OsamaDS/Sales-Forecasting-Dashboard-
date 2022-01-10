import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for
from flask_cors import cross_origin
from model import model
import pickle
import os


PEOPLE_FOLDER = os.path.join('static', 'img')

df = pd.read_csv('data.csv')

labels = list(df['Order weekname'])
values = list(df['Sales'])



print('labels = ', labels[:5])
print('values = ', values[:5])


model_ = model()
stats_test = model_.train(df)

with open("learned_model.pkl", 'rb') as f:
    lr = pickle.load(f)


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

@app.route('/')
@cross_origin()
def index():
    l_labels, l_values = Line(df)
    p_labels, p_values = Pie(df)
    b_labels, b_values = Bar(df)
    data = {'Line': {'labels': l_labels, 'values': l_values} ,
     'Pie': {'labels': p_labels, 'values': p_values},
     'Bar': {'labels': b_labels, 'values': b_values}
    }
    
    t_headings, t_values = customer_table(df)

    total_revenue = str(int(np.round(df['Sales'].sum(), 0))) + '$'
    total_sales = df['Sales'].count()
    total_customers = len(df['Customer Name'].value_counts())

    return render_template("index.html", data=data, headings = t_headings, values = t_values, total_revenue=total_revenue,
                                total_sales=total_sales, total_customers=total_customers)

@app.route('/charts')
@cross_origin()
def charts():
    b_labels, b_values = ver_bar(df)
    p_labels, p_values = ver_pie(df)
    m_labels, m_values = ver_bar_month(df)
    l_labels, central_values, east_values, west_values, south_values = ver_line(df)

    data = {'Pie': {'labels': p_labels, 'values': p_values},
     'Bar': {'labels': b_labels, 'values': b_values},
     'm_Bar': {'labels': m_labels, 'values': m_values},
     'Line': {'labels': l_labels, 'central_values': central_values, 
            'east_values': east_values, 'west_values': west_values, 'south_values': south_values}
    }

    return render_template('chart.html', data=data)

@app.route('/model', methods=['GET','POST'])
@cross_origin()
def model():
    filename = os.path.join(app.config['UPLOAD_FOLDER'], 'performance.png')
    image2 = os.path.join(app.config['UPLOAD_FOLDER'], 'rolling_stats.png')
    image3 = os.path.join(app.config['UPLOAD_FOLDER'], 'sales_prediction.png')

    # model_.train(df)

    dataset = pd.read_csv('sales.csv')
    error = model_.predict(lr, dataset, '2018-01-01','2018-12-30')
    return render_template('model.html', image1 = filename, image2=image2, error=error, image3=image3, stats_test=stats_test)



def Bar(df):
    df.sort_values(by=['Order weekday'], inplace=True)
    df1 = pd.DataFrame(df.groupby(['Order weekday','Order weekname'])['Sales'].sum())
    df1.reset_index(level=0, drop=True, inplace=True)
    df1.reset_index(inplace=True)
    labels = list(df1['Order weekname'])
    values = list(df1['Sales'])
    return labels, values

def Pie(df):
    df1 = pd.DataFrame(df.groupby(['Segment'])['Sales'].sum())
    df1.reset_index(inplace=True)
    df1['Sales Percent'] = np.round((df1['Sales']/df1['Sales'].sum()) * 100)
    labels = list(df1['Segment'])
    values = list(df1['Sales Percent'])
    return labels, values

def Line(df):
    df1 = pd.DataFrame(df.groupby(['Order Year'])['Sales'].sum())
    df1.reset_index(inplace=True)
    labels = list(df1['Order Year'])
    values = list(df1['Sales'])
    return labels, values

def customer_table(df):
    df1 = pd.DataFrame(df.groupby(['Customer ID', 'Customer Name'])['Sales'].sum())
    df1.reset_index(inplace=True)
    df1 = df1.sort_values('Sales', ascending=False)[:10]
    df1['Sales'] = np.round(df1['Sales'],0)
    df1['Sales'] = df1['Sales'].astype('int')
    df1['Sales'] = df1['Sales'].astype('str')
    df1['Sales'] = df1['Sales'].apply(lambda x: x+"$")
    df1.reset_index(drop=True, inplace=True)
    headings = list(df1.columns)
    values = df1.values.tolist()

    return headings, values

def ver_bar(df):
    df1 = pd.DataFrame(df.groupby('State')['Sales'].sum())
    df1.reset_index(inplace=True)
    df1.sort_values('Sales',ascending=False, inplace=True)
    df1 = df1.reset_index(drop=True)[:15]
    df1['Sales'] = np.round(df1['Sales'],2)
    labels = list(df1['State'])
    values = list(df1['Sales'])
    return labels, values

def ver_pie(df):
    df1 = pd.DataFrame(df.groupby(['Category'])['Sales'].sum())
    df1.reset_index(inplace=True)
    df1['Sales Percent'] = np.round((df1['Sales']/df1['Sales'].sum()) * 100)
    labels = list(df1['Category'])
    values = list(df1['Sales Percent'])
    return labels, values

def ver_bar_month(df):
    df1 = pd.DataFrame(df.groupby(['Order Month'])['Sales'].sum())
    df1.reset_index(inplace=True)
    months_num = [4,8,12,2,1,7,6,3,5,11,10,9]
    df1['months_num'] = months_num
    df1.sort_values('months_num', inplace=True)
    labels = list(df1['Order Month'])
    values = list(df1['Sales'])
    return labels, values

def ver_line(df):
    df_central = pd.DataFrame(df[df['Region']=='Central'].groupby('Order Year')['Sales'].sum())
    df_east = pd.DataFrame(df[df['Region']=='East'].groupby('Order Year')['Sales'].sum())
    df_south = pd.DataFrame(df[df['Region']=='South'].groupby('Order Year')['Sales'].sum())
    df_west = pd.DataFrame(df[df['Region']=='West'].groupby('Order Year')['Sales'].sum())

    df_central.reset_index(inplace=True)
    df_east.reset_index(inplace=True)
    df_south.reset_index(inplace=True)
    df_west.reset_index(inplace=True)

    east_sales = list(df_east['Sales'])
    south_sales = list(df_south['Sales'])
    west_sales = list(df_west['Sales'])

    df_central['east_sales'] = east_sales
    df_central['west_sales'] = west_sales
    df_central['south_sales'] = south_sales

    labels = list(df_central['Order Year'])
    central_values = list(df_central['Sales'])
    east_values = list(df_central['east_sales'])
    west_values = list(df_central['west_sales'])
    south_values = list(df_central['south_sales'])

    return labels, central_values, east_values, west_values, south_values


if __name__=='__main__':
    app.run()