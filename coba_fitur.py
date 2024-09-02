import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import re
st.set_page_config(layout="wide")

#Data Source Section
@st.cache_data
def load_data(url):
    data = pd.read_csv(url)
    return(data)

# df1=load_data(r"CONVERT JUBELIO TO DASHBOARD - Daftar Penjualan Barang.csv")
# df2=load_data(r"Copy of CONVERT JUBELIO TO DASHBOARD - Daftar Penjualan Barang.csv")
# df3=load_data(r"CONVERT JUBELIO TO DASHBOARD - Daftar Penjualan Barang(7).csv")

df1 = load_data(r"jan-april_2024.csv")
df2 = load_data(r"mei-juni_2024.csv")
df2 = df2.rename(columns={'Harga Setelah Diskon': 'amount'})
df3 = load_data(r"Juli-agustus_2024.csv")
dfsopi = load_data(r'dfsopi.csv')
# #Data Source Section Finish


#Dataframe Section
# @st.cache_data
# def cleaning_data(df):
#     # df['Username'] = df['Pelanggan']+df['No Telp']
#     # df['Date'] = pd.to_datetime(df['Tanggal'], errors='coerce')
#     # df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M:%S')
#     # df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
#     # df['Date'] = pd.to_datetime(df['Date'])
#     df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d/%b/%Y %H:%M')
#     df['Username'] = df['Pelanggan']+df['No Telp']
#     df['Date'] = df['Tanggal'].dt.strftime('%Y-%m-%d')
#     df['Date'] = pd.to_datetime(df['Date'])
#     df = df.loc[df['Username'].str.contains('barudak nsi') == False]
#     df = df.loc[df['Username'].str.contains('Barudak NSI') == False]
#     df = df.loc[df['Username'].str.contains('cod') == False]
#     df = df.loc[df['Username'].str.contains('Pelanggan Umum') == False]
#     df = df.loc[df['Username'].str.contains('orderan pa yanto') == False]
#     df = df.loc[df['Username'].str.contains('Rani Fitriank6281284956014') == False]
#     df = df.loc[df['Sumber'].str.contains('INTERNAL') == False]
#     df = df.loc[df['Nama Barang'].str.contains('Unique Code') == False]
#     return(df)

# def cleaning_data2(df):
#     # df['Username'] = df['Pelanggan']+df['No Telp']
#     # df['Date'] = pd.to_datetime(df['Tanggal'], errors='coerce')
#     # df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M:%S')
#     # df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
#     # df['Date'] = pd.to_datetime(df['Date'])
#     df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d/%b/%Y')
#     df['Username'] = df['Pelanggan']+df['No Telp']
#     df['Date'] = df['Tanggal'].dt.strftime('%Y-%m-%d')
#     df['Date'] = pd.to_datetime(df['Date'])
#     df = df.loc[df['Username'].str.contains('barudak nsi') == False]
#     df = df.loc[df['Username'].str.contains('Barudak NSI') == False]
#     df = df.loc[df['Username'].str.contains('cod') == False]
#     df = df.loc[df['Username'].str.contains('Pelanggan Umum') == False]
#     df = df.loc[df['Username'].str.contains('orderan pa yanto') == False]
#     df = df.loc[df['Username'].str.contains('Rani Fitriank6281284956014') == False]
#     df = df.loc[df['Sumber'].str.contains('INTERNAL') == False]
#     df = df.loc[df['Nama Barang'].str.contains('Unique Code') == False]
#     return(df)

# df1 = cleaning_data(df1)
# df1 = df1.loc[(df1['Date']>='2024-01-01')&(df1['Date']<='2024-04-30')]
# df2 = cleaning_data2(df2)
# df2 = df2.loc[(df2['Date']>='2024-05-01')&(df2['Date']<='2024-06-30')]
# df3 = cleaning_data(df3)
# df3 = df3.loc[(df3['Date']>='2024-07-01')&(df3['Date']<='2024-08-31')]

# dfjb = pd.concat([df1,df2,df3])
# dfjb = dfjb.loc[dfjb['Sumber']!='SHOPEE']

# # Df Shopee
# dfsopi = dfsopi.rename(columns={'Waktu Pembayaran Dilakukan': 'Tanggal','Harga Setelah Diskon': 'amount','No. Pesanan':'No Pesanan','Username (Pembeli)':'Pelanggan','No. Telepon':'No Telp','Jumlah':'QTY','Nomor Referensi SKU':'SKU','Nama Produk':'Nama Barang'})
# dfsopi['Tanggal'] = pd.to_datetime(dfsopi['Tanggal'], format='%Y-%m-%d %H:%M')
# dfsopi['No Telp'] = dfsopi['No Telp'].apply(str)
# dfsopi['Username'] = dfsopi['Pelanggan']+dfsopi['No Telp']
# dfsopi['Date'] = dfsopi['Tanggal'].dt.strftime('%Y-%m-%d')
# dfsopi['No Pesanan']= 'SP-'+dfsopi['No Pesanan']

# dfz = pd.concat([dfjb,dfsopi])
# dfz['amount'] = pd.to_numeric(dfz['amount'], errors='coerce')
# dfz = dfz.loc[dfz['amount']>0]
# dfz['Qyear'] = pd.PeriodIndex(dfz.Date,freq="Q")
# 

df1['Tanggal'] = pd.to_datetime(df1['Tanggal'], format='%d/%b/%Y %H:%M')
df1['Username'] = df1['Pelanggan']+df1['No Telp']
df1['Date'] = df1['Tanggal'].dt.strftime('%Y-%m-%d')
df1['Date'] = pd.to_datetime(df1['Date'])
df1 = df1.loc[(df1['Date']>='2024-01-01')&(df1['Date']<='2024-04-30')]

df1 = df1.loc[df1['Username'].str.contains('barudak nsi') == False]
df1 = df1.loc[df1['Username'].str.contains('Barudak NSI') == False]
df1 = df1.loc[df1['Username'].str.contains('cod') == False]
df1 = df1.loc[df1['Username'].str.contains('Pelanggan Umum') == False]
df1 = df1.loc[df1['Username'].str.contains('orderan pa yanto') == False]
df1 = df1.loc[df1['Username'].str.contains('Rani Fitriank6281284956014') == False]
df1 = df1.loc[df1['Nama Barang'].str.contains('Unique Code') == False]
#  filter jan-april

df2['Tanggal'] = pd.to_datetime(df2['Tanggal'], format='%d/%b/%Y')
df2['Username'] = df2['Pelanggan']+df2['No Telp']
df2['Date'] = df2['Tanggal'].dt.strftime('%Y-%m-%d')
df2['Date'] = pd.to_datetime(df2['Date'])
df2 = df2.loc[(df2['Date']>='2024-05-01 00:00:00')&(df2['Date']<='2024-06-30 00:00:00')]

df2 = df2.loc[df2['Username'].str.contains('barudak nsi') == False]
df2 = df2.loc[df2['Username'].str.contains('Barudak NSI') == False]
df2 = df2.loc[df2['Username'].str.contains('cod') == False]
df2 = df2.loc[df2['Username'].str.contains('Pelanggan Umum') == False]
df2 = df2.loc[df2['Username'].str.contains('orderan pa yanto') == False]
df2 = df2.loc[df2['Username'].str.contains('Rani Fitriank6281284956014') == False]
df2 = df2.loc[df2['Nama Barang'].str.contains('Unique Code') == False]
# filter mei-juni

df3['Tanggal'] = pd.to_datetime(df3['Tanggal'], format='%d/%b/%Y %H:%M')
df3['Username'] = df3['Pelanggan']+df3['No Telp']
df3['Date'] = df3['Tanggal'].dt.strftime('%Y-%m-%d')
df3['Date'] = pd.to_datetime(df3['Date'])
df3 = df3.loc[(df3['Date']>='2024-07-01')&(df3['Date']<='2024-08-31')]

df3 = df3.loc[df3['Username'].str.contains('barudak nsi') == False]
df3 = df3.loc[df3['Username'].str.contains('Barudak NSI') == False]
df3 = df3.loc[df3['Username'].str.contains('cod') == False]
df3 = df3.loc[df3['Username'].str.contains('Pelanggan Umum') == False]
df3 = df3.loc[df3['Username'].str.contains('orderan pa yanto') == False]
df3 = df3.loc[df3['Username'].str.contains('Rani Fitriank6281284956014') == False]
df3 = df3.loc[df3['Nama Barang'].str.contains('Unique Code') == False]
#  filter juli-agustus

dfjb = pd.concat([df1,df2,df3])
dfjb = dfjb.loc[dfjb['Sumber']!='SHOPEE']

# Df Shopee
dfsopi = dfsopi.rename(columns={'Waktu Pembayaran Dilakukan': 'Tanggal','Harga Setelah Diskon': 'amount','No. Pesanan':'No Pesanan','Username (Pembeli)':'Pelanggan','No. Telepon':'No Telp','Jumlah':'QTY','Nomor Referensi SKU':'SKU','Nama Produk':'Nama Barang'})
dfsopi['Tanggal'] = pd.to_datetime(dfsopi['Tanggal'], format='%Y-%m-%d %H:%M')
dfsopi['No Telp'] = dfsopi['No Telp'].apply(str)
dfsopi['Username'] = dfsopi['Pelanggan']+dfsopi['No Telp']
dfsopi['Date'] = dfsopi['Tanggal'].dt.strftime('%Y-%m-%d')
dfsopi['No Pesanan']= 'SP-'+dfsopi['No Pesanan']

dfz = pd.concat([dfjb,dfsopi])
dfz['amount'] = pd.to_numeric(dfz['amount'], errors='coerce')
dfz = dfz.loc[dfz['amount']>0]
dfz['Qyear'] = pd.PeriodIndex(dfz.Date,freq="Q")



# dfx = df1.copy()
# dfy = df2.copy()
# dfw = df3.copy()

# dfw = dfw.rename(columns={'Harga Setelah Diskon': 'amount'})
# dfx = dfx.loc[dfx['amount']>99000]
# dfy = dfy.loc[dfy['amount']>99000]
# dfw = dfw.loc[dfw['amount']>99000]
# dfx['Qyear'] = pd.PeriodIndex(dfx.Date,freq="Q")
# dfy['Qyear'] = pd.PeriodIndex(dfy.Date,freq="Q")
# dfw['Qyear'] = pd.PeriodIndex(dfw.Date,freq="Q")
# dfx = dfz.loc[(dfz['Qyear']=='2024Q1')]
# dfy = dfz.loc[(dfz['Qyear']=='2023Q4')]
# dfw = dfz.loc[(dfz['Qyear']=='2024Q2')]
# dfz = pd.concat([dfx,dfy,dfw])

# dfz['Month'] = dfz['Date'].dt.month
# dfz['Year'] = dfz['Date'].dt.year
# dfz['dateInt']=dfz['Year'].astype(str) + dfz['Month'].astype(str).str.zfill(2)
# dfz['year_month'] = pd.to_datetime(dfz['dateInt'], format='%Y%m')
dfz = dfz.loc[dfz['amount']>99000]
# dfz_rfm = dfz.loc[dfz['Date']>'31-12-2023']


max = dfz.groupby(['Username']).Date.max().reset_index()
min = dfz.groupby(['Username']).Date.min().reset_index()
fr = dfz.groupby(['Username'])['No Pesanan'].nunique().reset_index()

dfzz = pd.merge(min,max, how='outer', on=['Username'])
dfzz = pd.merge(fr,dfzz, how='outer', on=['Username'])

dfzz['Qyear1'] = pd.PeriodIndex(dfzz.Date_x,freq="Q")
dfzz['Qyear2'] = pd.PeriodIndex(dfzz.Date_y,freq="Q")

ren = dfzz.loc[dfzz['No Pesanan']>1]
ren1 = ren.loc[(ren['Qyear1']=='2024Q2')&(ren['Qyear2']=='2024Q3')]
ren2 = ren.loc[(ren['Qyear1']=='2024Q3')&(ren['Qyear2']=='2024Q3')]
ren3 = ren.loc[(ren['Qyear1']=='2024Q2')&(ren['Qyear2']=='2024Q2')]
#Dataframe Section Finish

#Variabel & process data
tabel_unique_costumer_q32024=dfz.loc[dfz['Qyear']=='2024Q3']['Username'].nunique()
tabel_unique_costumer_q22024=dfz.loc[dfz['Qyear']=='2024Q2']['Username'].nunique()
delta_unique_coustumer = tabel_unique_costumer_q32024-tabel_unique_costumer_q22024
new_costumer_q32024 = dfzz.loc[dfzz['Qyear1']=='2024Q3']['Username'].count()
retention_costumer_q32024 = ren1['Username'].count()+ren2['Username'].count()

avg_net_sales_q32024 = dfz.loc[dfz['Qyear']=='2024Q3']['amount'].mean()
avg_frequency_q32024 = dfz.loc[dfz['Qyear']=='2024Q3'].groupby(['Username'])['No Pesanan'].nunique().mean()
tot_gross_q32024 = dfz.loc[dfz['Qyear']=='2024Q3']['amount'].sum()
tot_costumer_q32024 = dfz.loc[dfz['Qyear']=='2024Q3']['Username'].count()
clv_q32024 = round(tot_gross_q32024/tot_costumer_q32024*avg_frequency_q32024,0)
clv_q32024v = f'Rp.{clv_q32024}'

avg_net_sales_q22024 = round(dfz.loc[dfz['Qyear']=='2024Q2']['amount'].mean(),0)
avg_net_sales_q22024v = f'Rp.{avg_net_sales_q22024}'

avg_frequency_q22024 = dfz.loc[dfz['Qyear']=='2024Q2'].groupby(['Username'])['No Pesanan'].nunique().mean()

tot_gross_q22024 = dfz.loc[dfz['Qyear']=='2024Q2']['amount'].sum()
tot_costumer_q22024 = dfz.loc[dfz['Qyear']=='2024Q2']['Username'].count()
clv_q22024 = tot_gross_q22024/tot_costumer_q22024*avg_frequency_q22024


delta_avg_net_sales= avg_net_sales_q32024-avg_net_sales_q22024
delta_avg_frequency= avg_frequency_q32024-avg_frequency_q22024
delta_clv= clv_q32024-clv_q22024
#Variabel & process data Finish

# RFM data Proses

@st.cache_data

def rfm(dfz):
    recency = dfz.groupby(['Username']).Date.max().reset_index()
    recency['Recency'] = recency['Date'] - pd.Timestamp.today()
    recency['Recency'] = recency['Recency'].dt.days
    recency['Recency'] = recency['Recency']*-1
    frequncy = dfz.groupby(['Username'])['No Pesanan'].nunique().reset_index()
    monetary = dfz.groupby(['Username'])['amount'].sum().reset_index()
    m1 = pd.merge(recency,frequncy, how='outer', on=['Username'])
    m2 = pd.merge(m1,monetary, how='outer', on=['Username'])
    mss  = m2[['Username','Recency','No Pesanan','amount']]

    mf = mss.copy()
    x = mf[['Recency', 'No Pesanan', 'amount']]

    scaledX = preprocessing.minmax_scale(x,feature_range=(0,5))

    scaledX = pd.DataFrame(scaledX)
    scaledX.columns =['Recency_scale', 'No Pesanan_scale', 'amount_scale']
    mf = pd.concat([mf,scaledX], axis=1)
    mf['Recency_scale'] = mf['Recency_scale'].max()-mf['Recency_scale']
    mf['value_rfm'] = mf['Recency_scale']+mf['No Pesanan_scale']+mf['amount_scale']
    mf['rank_rfm'] =pd.qcut(mf['value_rfm'], q=[0, .20, .40, .60, .80, 1], labels=['1', '2', '3','4','5'])

    mf.loc[((mf['rank_rfm'] == '5')|(mf['rank_rfm'] == '2')|(mf['rank_rfm'] == '3')|(mf['rank_rfm'] == '4'))  & (mf['amount'] < 1000000), 'Cluster'] = 'Potential Costumer'
    mf.loc[(mf['rank_rfm'] == '5') & (mf['Recency'] < 31) & (mf['No Pesanan'] == 1), 'Cluster'] = 'New Costumer' 
    mf.loc[((mf['rank_rfm'] == '5')|(mf['rank_rfm'] == '2')|(mf['rank_rfm'] == '3')|(mf['rank_rfm'] == '4')) & (mf['No Pesanan'] == 1) & (mf['amount'] > 1000000), 'Cluster'] = 'Potential Loyal' 
    mf.loc[((mf['rank_rfm'] == '5')|(mf['rank_rfm'] == '2')|(mf['rank_rfm'] == '3')|(mf['rank_rfm'] == '4')) & (mf['No Pesanan'] > 1) & (mf['amount'] > 1000000), 'Cluster'] = 'Loyal Costumer'

    mf.loc[(mf['rank_rfm'] == '1'), 'Cluster'] = 'Lost Costumer'
    mf = mf[['Username','Recency', 'No Pesanan', 'amount','Cluster']]
    mf.columns = ['Username','Recency', 'Frequency', 'Monetary','Cluster']
    mf  = pd.merge(mf,dfz,on=['Username'], how='left')
    mf = mf.drop_duplicates(subset=['Username'])
    mf = mf[['Username', 'Pelanggan' , 'No Telp' ,'Recency', 'Frequency', 'Monetary' ,'Cluster']]

    return mf

mf = rfm(dfz)

cluster = pd.DataFrame(mf['Cluster'].value_counts().reset_index())

# RFM data proses finish

#Interface section
st.write("""
         # CRM DASHBOARD
         """)
st.write("##### Update Monthly")



container = st.container(border=True)
col1, col2, col3 = st.columns(3)

with col1:
    col1.metric(label='Total Customer unique Q3 2024', value=tabel_unique_costumer_q32024, delta=delta_unique_coustumer)
with col2:
    col2.metric(label='Total New Customer Q3 2024', value=new_costumer_q32024)
with col3:
    col3.metric(label='Total Loyal Retention Customer Q3 2024',value=retention_costumer_q32024)


st.write("""
         ## CLV
         """)

col4, col5, col6 = st.columns(3)
with col4:
    col4.metric(label='Avg Net Sales Q3', value=avg_net_sales_q22024v, delta=round(delta_avg_net_sales,0))
with col5:
    col5.metric(label='Avg Frequency Q3 2024', value=round(avg_frequency_q32024,2), delta=round(delta_avg_frequency,2))
with col6:
    col6.metric(label='CVL Value',value=clv_q32024v, delta=round(delta_clv,0))


# fig, ax = plt.subplots()
# ax.hist(data=dfz.groupby(['Date'])['amount'].sum(), x='Date',y='amount')

# st.pyplot(fig)
st.write("#")

st.write("# --------------------RFM analyst-------------------")

st.write("## Clustering Customer 2024")

fig1 = plt.figure(figsize=(5, 2))
ax = sns.barplot(cluster, x="Cluster", y="count", estimator="sum", errorbar=None)
ax.bar_label(ax.containers[0], fontsize=10)
plt.xticks(rotation=15)

fig2 = plt.figure(figsize=(10   , 4))
sns.scatterplot(data=mf,x='Recency',y='Monetary',hue='Cluster')

col7, col8 = st.columns(2)
with col7:
    col7.dataframe(cluster)
    col7.write('Loyal Costumer : pembel yang sudah melakukan pembelian > 1 dan nilai transaksi > Rp.1.000.000')
    col7.write('Petential Loyal : pembel yang sudah melakukan pembelian == 1 dan nilai transaksi > Rp.1.000.000')
    col7.write('Petential Costumer : pembeli dengan total nilai transaksi < Rp.1.000.000')
    col7.write('New Costumer : pembel yang sudah melakukan pembelian = 1 dan dengan waktu 30 hari trakhir')
    col7.write('Lost Costumer : pembel yang sudah melakukan pembelian 160 hari yang lalu')

with col8:
    # col8.bar_chart(cluster,x="Cluster", y="count", color="Cluster")
    col8.pyplot(fig1)
    # col8.pyplot(fig2)
st.write("##")
st.pyplot(fig2)
st.write("##")
on = st.toggle("Switch on for executable data")
if on:
    BAD_CHARS = ['*']
    pat = '|'.join(['({})'.format(re.escape(c)) for c in BAD_CHARS])

    mf = mf[~mf['Username'].str.contains(pat)]

list_cluster = mf['Cluster'].unique()
options = st.multiselect("What are your favorite colors",list_cluster,list_cluster)
mf = mf.loc[mf['Cluster'].isin(options)]
st.dataframe(mf)












