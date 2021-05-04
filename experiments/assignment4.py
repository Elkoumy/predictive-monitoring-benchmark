import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib


def get_pos_case_length_quantile(data, quantile=0.90):
    return int(np.ceil(data.groupby('Case ID').size().quantile(quantile)))

# split into training and test
def split_data_strict(data, train_ratio, split="temporal"):
    # split into train and test using temporal split and discard events that overlap the periods
    data = data.sort_values(['time:timestamp', 'Activity'], ascending=True, kind='mergesort')
    grouped = data.groupby('Case ID')
    start_timestamps = grouped['time:timestamp'].min().reset_index()
    start_timestamps = start_timestamps.sort_values('time:timestamp', ascending=True, kind='mergesort')
    train_ids = list(start_timestamps['Case ID'])[:int(train_ratio*len(start_timestamps))]
    train = data[data['Case ID'].isin(train_ids)].sort_values(['time:timestamp', 'Activity'], ascending=True, kind='mergesort')
    test = data[~data['Case ID'].isin(train_ids)].sort_values(['time:timestamp', 'Activity'], ascending=True, kind='mergesort')
    split_ts = test['time:timestamp'].min()
    train = train[train['time:timestamp'] < split_ts]
    return (train, test)


from sklearn.model_selection import StratifiedKFold
def get_stratified_split_generator(data, n_splits=5, shuffle=True, random_state=22):
    grouped_firsts = data.groupby('Case ID', as_index=False).first()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    for train_index, test_index in skf.split(grouped_firsts, grouped_firsts['label']):
        current_train_names = grouped_firsts['Case ID'][train_index]
        train_chunk = data[data['Case ID'].isin(current_train_names)].sort_values('time:timestamp', ascending=True, kind='mergesort')
        test_chunk = data[~data['Case ID'].isin(current_train_names)].sort_values('time:timestamp', ascending=True, kind='mergesort')
        yield (train_chunk, test_chunk)

def get_class_ratio(data):
    class_freqs = data['label'].value_counts()
    return class_freqs[1] / class_freqs.sum()


# GAP parameter
def generate_prefix_data_old(data, min_length, max_length, gap=1):
    # generate prefix data (each possible prefix becomes a trace)
    data['case_length'] = data.groupby('Case ID')['Activity'].transform(len)

    dt_prefixes = data[data['case_length'] >= min_length].groupby('Case ID').head(min_length)
    dt_prefixes["prefix_nr"] = 1
    dt_prefixes["orig_case_id"] = dt_prefixes['Case ID']
    for nr_events in range(min_length+gap, max_length+1, gap):
        tmp = data[data['case_length'] >= nr_events].groupby('Case ID').head(nr_events)
        tmp["orig_case_id"] = tmp['Case ID']
        tmp['Case ID'] = tmp['Case ID'].apply(lambda x: "%s_%s"%(x, nr_events))
        tmp["prefix_nr"] = nr_events
        dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)

    dt_prefixes['case_length'] = dt_prefixes['case_length'].apply(lambda x: min(max_length, x))

    return dt_prefixes


def generate_prefix_data(data, ngram_size):
    # generate prefix data (each possible prefix becomes a trace)

    # ngram_size=3
    dt_prefixes=data.groupby(['Case ID']).apply(create_ngrams, ngram_size)

    dt_prefixes=dt_prefixes.rename(columns={'Case ID': 'newcaseid'})
    dt_prefixes=dt_prefixes.reset_index().rename(columns={'Case ID': 'original_caseid'})
    dt_prefixes=dt_prefixes.drop('level_1',axis=1)
    dt_prefixes=dt_prefixes.rename(columns={'newcaseid': 'Case ID'})

    return dt_prefixes


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

        prefix['Case ID']=prefix['Case ID']+'_'+str(idx)
        prefix['prefix_nr'] = idx + 1
        result=pd.concat([result,prefix])

    return result

df=pd.read_csv(r'C:\Gamal Elkoumy\PhD\OneDrive - Tartu Ülikool\Courses\Process Mining\Assignment4\predictive-monitoring-benchmark\data\turnaround_anon_sla.csv')

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


df=df.rename(columns={'caseid': 'Case ID','activity':'Activity', 'start_timestamp':'time:timestamp'})
df.to_csv(r'C:\Gamal Elkoumy\PhD\OneDrive - Tartu Ülikool\Courses\Process Mining\Assignment4\predictive-monitoring-benchmark\experiments\experiment_log\turnaround_anon_sla_renamed.csv',index=False, sep=';')

"""Split into train and test"""
train_ratio = 0.8
n_splits = 2
random_state = 22

max_prefix_length = min(40, get_pos_case_length_quantile(df, 0.90))

train, test = split_data_strict(df, train_ratio, split="temporal")


"""Q3"""


# prepare chunks for CV
dt_prefixes = []
class_ratios = []
min_prefix_length = 1
ngram_size=5

for train_chunk, test_chunk in get_stratified_split_generator(train, n_splits=n_splits):
    class_ratios.append(get_class_ratio(train_chunk))
    # generate data where each prefix is a separate instance
    dt_prefixes.append(generate_prefix_data(test_chunk, ngram_size))
del train

"""Q4"""


import BucketFactory

# encoding_method = "last", "agg", "index"
# Bucketing prefixes based on control flow
bucketer_args = {'encoding_method': 'last',
                 'case_id_col': 'Case ID',
                 'cat_cols':['Activity'],
                 'num_cols':[],
                 'random_state':random_state}

cv_iter = 0
dt_test_prefixes = dt_prefixes[cv_iter]
dt_train_prefixes = pd.DataFrame()
for cv_train_iter in range(n_splits):
    if cv_train_iter != cv_iter:
        dt_train_prefixes = pd.concat([dt_train_prefixes, dt_prefixes[cv_train_iter]], axis=0)


cv_iter = 0
dt_test_prefixes = dt_prefixes[cv_iter]
dt_train_prefixes = pd.DataFrame()
for cv_train_iter in range(n_splits):
    if cv_train_iter != cv_iter:
        dt_train_prefixes = pd.concat([dt_train_prefixes, dt_prefixes[cv_train_iter]], axis=0)


""" ************* Performing Cluster Bucketing ***************"""

#bucket_methods = "single", "prefix", "state", "cluster", "knn"
bucket_method = 'cluster'
if bucket_method == "cluster":
    bucketer_args["n_clusters"] = 3
bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)
bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)
bucket_assignments_test = bucketer.predict(dt_test_prefixes)

""" Train buckets"""
bucket_number = 2
bucket_indexes = dt_train_prefixes.groupby('Case ID').first().index
bucket_indexes = bucket_indexes[bucket_assignments_train == bucket_number]
print(bucket_indexes)
bucket_data = dt_train_prefixes[dt_train_prefixes['Case ID'].isin(bucket_indexes)]

def get_label_numeric(data):
    y = data.groupby('Case ID').first()['label'] # one row per case
    return y
train_y = get_label_numeric(bucket_data)




"""Test Buckets"""

bucket_indexes = dt_test_prefixes.groupby('Case ID').first().index
bucket_indexes = bucket_indexes[bucket_assignments_test == bucket_number]
bucket_data_test = dt_test_prefixes[dt_test_prefixes['Case ID'].isin(bucket_indexes)]
test_y = get_label_numeric(bucket_data_test)

""" ************* Perfroming Index Encoding ******************"""
import EncoderFactory
from sklearn.pipeline import FeatureUnion, Pipeline

cls_encoder_args = {'case_id_col': 'Case ID',
                    'static_cat_cols': [],
                    'static_num_cols': [],
                    'dynamic_cat_cols': ['Activity'],
                    'dynamic_num_cols': ["WIP"],
                    'fillna': True}

encoding_dict = {
    "laststate": ["static", "last"],
    "agg": ["static", "agg"],
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"]
}

methods = encoding_dict['index']

feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])

encoding = feature_combiner.fit_transform(bucket_data, train_y)

pd.DataFrame(encoding).to_csv('encoding.csv')

"""******* Perfroming training **************"""

import xgboost as xgb



model_parameters=pd.read_pickle(r'C:\Gamal Elkoumy\PhD\OneDrive - Tartu Ülikool\Courses\Process Mining\Assignment4\predictive-monitoring-benchmark\experiments\optimizer_log\optimal_params_xgboost_turnaround_anon_sla_renamed_cluster_index.pickle')

model= xgb.XGBClassifier(objective='binary:logistic',
                                    n_estimators=500,
    **model_parameters)


pipeline = Pipeline([('encoder', feature_combiner), ('cls', model)])
pipeline.fit(bucket_data, train_y)

preds_pos_label_idx = np.where(model.classes_ == 1)[0][0]
preds = pipeline.predict_proba(bucket_data_test)[:,preds_pos_label_idx]

from sklearn.metrics import roc_auc_score
score = roc_auc_score(test_y, preds)
print("The ROC AUC is : %s"%(score))
