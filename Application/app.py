import streamlit as st
import pickle
import sklearn
import numpy as np

#Defining path for pickle files


#Loading pickle fiels
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

#Creating frontend using streamlit
st.title("Laptop Predictor")
#Fields required
company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('Ram[in GB]', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of laptop')
touchscreen = st.selectbox('Touchscreen',['No', 'Yes'])
ips = st.selectbox('IPS',['No', 'Yes'])
clockspeed = st.number_input("Enter clockspeed")
ssd = st.selectbox('SSD[in GB]', [0, 8, 128, 256, 512, 1024])
hdd = st.selectbox('HDD[in GB]', [0, 128, 256, 512, 1024, 2048])
screen_size = st.number_input('Screen Size')
resolution = st.selectbox('Screen Resolution', ['1920x1080','1366x768','1600x900','3840x2160','2880x1800','2560x1600','2560x1440','2304x1440'])
gpu = st.selectbox('GPU', df['GpuBrand'].unique())
os = st.selectbox('OS', df['OS'].unique())

if st.button('Predict Price'):

    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else: 
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,weight,touchscreen,ips,clockspeed,ssd,hdd,ppi,gpu,os], dtype = object)

    query = query.reshape(1,12)
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))

