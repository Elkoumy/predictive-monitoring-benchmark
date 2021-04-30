import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

df=pd.read_csv('data/turnaround_anon_sla.csv')

#converting datatypes , timestamps
df.start_timestamp= pd.to_datetime(df.start_timestamp,utc=True)
df.end_timestamp= pd.to_datetime(df.end_timestamp,utc=True)

#calculating the start time and end time of every case
df['end_time']=df.groupby(['caseid']).end_timestamp.transform('max')
df['start_time']=df.groupby(['caseid']).start_timestamp.transform('min')

#calculating case duration in minutes ( the same time unit as the SLA)
df['duration']=(df.end_time-df.start_time).astype('timedelta64[m]')


#creating the label column
df['label']=1
df.loc[df.duration<=df['SLA MIN'],'label']=0
