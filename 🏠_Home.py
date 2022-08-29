import streamlit as st
import numpy as np
import pandas as pd
import pickle as pkl
import sklearn
import time
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest


isolation_model1 = IsolationForest(contamination = 0.03)

st.set_page_config(page_title="Kissan Pro",
                   page_icon="./app/images/kp.jpg", layout='wide', initial_sidebar_state="expanded")

st.title("Anomaly Detection")

header_cols = st.columns(3)
with header_cols[1]:
    header_cols[1].image('./images/download.jpg')

st.sidebar.info('Set the parameters to get the inference.')
index_level = st.sidebar.selectbox('Index_level', ["Rice", "Maize", "Chickpeas"], help="""Crop profiling""")


seg_res = st.sidebar.checkbox('Show Segmented Results', True)

cols = st.columns(3)

with cols[0]:
        x = int(cols[0].number_input('N-value', 0, 100000, step=1,
                                     help="""Nitrogen content of the soil"""))

with cols[1]:
        y = int(cols[1].number_input('P-value', 0, 100000, step=1,
                                     help="""Phosphorous content of the soi"""))

with cols[2]:
    z = int(cols[2].number_input('K-value', 0, 100000, step=1,
                                 help="""Potassium content of the soi"""))

cols1 = st.columns(2)

with cols1[0]:
        tissue_thresh = cols1[0].slider('Sensitivity', 0.0, 1.0, step=0.01, value=0.35, help="""Sensitivity for the Isolation-forest algorithm""")

with cols[1]:
        confidence_level = cols1[1].slider('Anomaly', 0.001, 0.99, step=0.01, value=0.75, help="""Anomaly to introduce""")


def load(path):
    BestModel = open(path, 'rb')
    model = pkl.load(BestModel)
    return model

def load_data(path):
    BestModel = open(path, 'rb')
    model = pkl.load(BestModel)
    return model

a = st.sidebar.selectbox('Nutrient', ['N', 'P', 'K'])

sub = st.button('Get Inference')
if sub:
    if index_level == 'rice':
        path = 'assets/rice.pickle'

    elif index_level == 'maize':
        path = 'assets/maize.pickle'

    else:
        path = 'assets/chickpea.pickle'

    model = load(path)
    data = load_data('assets/npk_synthetic_data.pickle')
    data = np.concatenate((data, np.array([x, y, z]).reshape(1, 3)), axis=0)

    isolation_model1.fit(data)

    x_, y_, z_ = 125, 65, 106

    with st.expander("Results", False):

        col11, col21, col31 = st.columns(3)
        col11.metric("Nitrogen", x, f"{(x - x_) / 100}%")
        col21.metric("Phosphorous", y, f"{(y - y_) / 100}%")
        col31.metric("Potassium", z, f"{(z - z_) / 100}%")

        preds = isolation_model1.predict(data)
        if preds[-1] == -1:
            st.error("Outlier detected!")
        else:
            st.success("Your farm is safe!")

        data = pd.DataFrame(data, columns=['N', 'P', 'K'])

        data['anomalies'] = preds
        anomalies1 = data.query('anomalies == -1')

        # plotting the scattered plot
        normal = go.Scatter(x=data.index.astype(str), y=data[a], name="Normal data", mode='markers')
        outlier = go.Scatter(x=anomalies1.index.astype(str), y=anomalies1[a], name="Anomalies", mode='markers',
                             marker=dict(color='red', size=5,
                                         line=dict(color='red', width=1)))

        # labelling
        layout = go.Layout(title="Isolation Forest", yaxis_title='Value', xaxis_title='x-axis', )

        # plotting
        Data = [normal, outlier]
        fig = go.Figure(data=Data, layout=layout)
        fig.show()






