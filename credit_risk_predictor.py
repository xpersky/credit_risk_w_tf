"""
This is my very first neural network so the full German Credit Data would be too difficult 
I want to make correlation between : age , credit amount, duration and credit risk
"""

# used imports

from sklearn.metrics import confusion_matrix, precision_score
from sklearn.model_selection import train_test_split

import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

import pandas as pd

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls

import numpy as np

# importing dataset

path = './custom_german_credit_dataset.csv'
data = pd.read_csv(path,index_col=0)

# Let's search some correlation between age and credit risk

risk_eq_good_by_ages = data.loc[data['Risk'] == 'good']['Age'].values.tolist()
risk_eq_bad__by_ages = data.loc[data['Risk'] == 'bad']['Age'].values.tolist()

# create histograms

trace1 = go.Histogram(
    x = risk_eq_good_by_ages,
    histnorm = 'probability',
    name = 'Credit Risk = Good'
)

trace2 = go.Histogram(
    x = risk_eq_bad__by_ages,
    histnorm = 'probability',
    name = 'Credit Risk = Bad'
)

# make graph

fig = tls.make_subplots(rows=2,cols = 1,shared_xaxes=False)
fig.append_trace(trace1,1,1)
fig.append_trace(trace2,2,1)
fig['layout'].update(showlegend=True,title='Credit Risk by Age Distribution', bargap=0.05)
py.plot(fig , filename = 'AgeDist', auto_open=True)

# Let's search some correlation between credit amount and credit risk

risk_eq_good_by_ca = data.loc[data['Risk'] == 'good']['Credit amount'].values.tolist()
risk_eq_bad__by_ca = data.loc[data['Risk'] == 'bad']['Credit amount'].values.tolist()
index_for_ca_riskG = data.loc[data['Risk'] == 'good']['Credit amount'].index.values.tolist()
index_for_ca_riskB = data.loc[data['Risk'] == 'bad']['Credit amount'].index.values.tolist()

# create lines

trace1 = go.Scatter(
    x = index_for_ca_riskG,
    y = risk_eq_good_by_ca,
    name = 'Good Credit Risk'
)

trace2 = go.Scatter(
    x = index_for_ca_riskB,
    y = risk_eq_bad__by_ca,
    name = 'Bad Credit Risk'
)

# make graph

graph = [trace1,trace2]

layout = dict(title = 'Credit Risk by Credit Amount Distribution')

fig = dict(data=graph,layout=layout)
py.plot(fig, filename='CADIST')

# Let's search some correlation between credit duration and credit risk

risk_eq_good_by_duration = data.loc[data['Risk'] == 'good']['Duration'].values.tolist()
risk_eq_bad__by_duration = data.loc[data['Risk'] == 'bad']['Duration'].values.tolist()

# create histograms

trace1 = go.Histogram(
    x = risk_eq_good_by_duration,
    histnorm = 'probability',
    name = 'Credit Risk = Good'
)

trace2 = go.Histogram(
    x = risk_eq_bad__by_duration,
    histnorm = 'probability',
    name = 'Credit Risk = Bad'
)

# make graph

fig = tls.make_subplots(rows=2,cols = 1,shared_xaxes=False)
fig.append_trace(trace1,1,1)
fig.append_trace(trace2,2,1)
fig['layout'].update(showlegend=True,title='Credit Risk by Duration Distribution', bargap=0.05)
py.plot(fig , filename = 'DurationDist', auto_open=True)

# let's normalize data

values = {'good' :1,'bad':0}
dataset = data.replace({'Risk' : values})
credit_max = max(dataset['Credit amount'].values.tolist())
age_max = max(dataset['Age'].values.tolist())
duration_max = max(dataset['Duration'].values.tolist())
dataset['Credit amount'] = dataset['Credit amount']/credit_max
dataset['Age'] = dataset['Age']/age_max
dataset['Duration'] = dataset['Duration']/duration_max
# lets make our model

x = dataset.drop(columns='Risk')
y = dataset['Risk']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

model = tf.keras.Sequential([
    layers.Dense(256,activation='relu',input_dim=3,kernel_regularizer=l2(0.01)),
    layers.Dropout(0.3, noise_shape=None,seed=None),
    layers.Dense(256,activation='relu',kernel_regularizer=l2(0.01)),
    layers.Dropout(0.3, noise_shape=None,seed=None),
    layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_out = model.fit(x_train,
                      y_train,
                      epochs=500,
                      batch_size=256,
                      verbose=1,
                      validation_data=(x_test,y_test))

print('Training acc ', np.mean(model_out.history['acc']))
print('Validation acc ', np.mean(model_out.history['val_acc'])) 

y_pred = model.predict(x_test)
rounded = [round(x[0]) for x in y_pred]
y_pred1 = np.array(rounded,dtype='int64')

print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))

model.save("CreditClassifier.h5")