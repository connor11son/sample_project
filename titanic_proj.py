import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

import pickle

# helper 
def fill_age(df):
    for i in range(1,4):
        df.loc[df["Pclass"] == i] = df.loc[df["Pclass"] == i].fillna(np.nanmedian(df.loc[df["Pclass"] == i]["Age"]))

def load_data(fname):
    df = pd.read_csv(fname)
    fill_age(df)
    df = df.drop("Cabin", axis=1)
    df = df.query("Embarked == Embarked")
    return df

st.title('Predicting Titanic Survivors')
st.image('titanic.jpg')

data = load_data('titanic_dset/train.csv')
st.dataframe(data.astype(str))

st.header('Look at these cool insights!')
males = data[data.Sex == "male"]

chart = alt.Chart(males).mark_bar().encode(alt.X('Age',type = 'ordinal', bin = True), 
                                                  alt.Y('count()', axis = alt.Axis(title = 'Number of Passengers')
                                                  ), color = 'Survived:N', column = 'Pclass',
                                                  tooltip = ['count()', 'Sex','Age', 'Fare'])
st.altair_chart(chart, use_container_width=False)

st.header('Run the Model on User Inputs!')
age = st.slider('Pick an Age', 0, 100)
fare = st.slider('Select a Fare', 0, 1000)
pclass = st.slider('Select a Passenger Class', 1, 3)
sex = st.selectbox('Choose the sex of the passenger', ['Male', 'Female'])
if sex == 'Male':
    male = 1
    female = 0
else:
    male = 0
    female = 1

passenger = np.array([age, fare, pclass, female, male]).reshape(1,-1)

with open('model.p', 'rb') as p:
    model = pickle.load(p)

res = np.round(model.predict_proba(passenger)[0][1]*100,2)

st.text('Your Passenger has a {}% Chance of Survival'.format(res))