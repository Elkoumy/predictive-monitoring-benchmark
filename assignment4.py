import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib


def count_wip(row, case_times):
    wip=0
    #started before start and ended after end
    #started after start and ended before end
    #started before start and ended before end
    #started before end and ended after end
    wip=case_times.loc[(case_times.case_start_time<= row.start_timestamp) & (case_times.case_end_time>=row.end_timestamp) |
                       (case_times.case_start_time >= row.start_timestamp) & (case_times.case_end_time <= row.end_timestamp)|
                       (case_times.case_start_time <= row.start_timestamp) & (case_times.case_end_time >= row.start_timestamp)|
                       (case_times.case_start_time <= row.end_timestamp) & (case_times.case_end_time >= row.end_timestamp)
                       ].shape[0]

    return wip


def create_ngrams(data, ngram_size):
    result=pd.DataFrame()


    for idx in range(0,data.shape[0]- ngram_size +1):

        prefix=data.iloc[idx:idx+ngram_size].copy()
        prefix=prefix.reset_index()

        prefix.caseid=prefix.caseid+'_'+str(idx)

        result=pd.concat([result,prefix])

    result=result.drop('index',axis=1)

    return result

df=pd.read_csv('data/turnaround_anon_sla.csv')

#converting datatypes , timestamps
df.start_timestamp= pd.to_datetime(df.start_timestamp,utc=True)
df.end_timestamp= pd.to_datetime(df.end_timestamp,utc=True)

df=df.sort_values(['start_timestamp']).reset_index()
df=df.drop('index',axis=1)

"""Q1"""

#calculating the start time and end time of every case
df['case_end_time']=df.groupby(['caseid']).end_timestamp.transform('max')
df['case_start_time']=df.groupby(['caseid']).start_timestamp.transform('min')

#calculating case duration in minutes ( the same time unit as the SLA)
df['duration']=(df.case_end_time-df.case_start_time).astype('timedelta64[m]')


#creating the label column
df['label']=1
df.loc[df.duration<=df['SLA MIN'],'label']=0
df.loc[df['SLA MIN'].isna(),'label']=0

"""Q2"""
case_times= pd.DataFrame()
case_times['case_end_time']=df.groupby(['caseid']).end_timestamp.max()
case_times['case_start_time']=df.groupby(['caseid']).start_timestamp.min()
case_times=case_times.reset_index()

df['WIP']=df.apply(count_wip,case_times=case_times ,axis=1)


"""Q3"""
ngram_size=3
result=df.groupby(['caseid']).apply(create_ngrams, ngram_size)
result=result.rename(columns={'caseid': 'newcaseid'})
result=result.reset_index().rename(columns={'caseid': 'original_caseid'})
result=result.drop('level_1',axis=1)
result=result.rename(columns={'newcaseid': 'caseid'})

