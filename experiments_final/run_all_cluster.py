import pandas as pd
import numpy as np
import sys
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cluster import KMeans
from time import time
import pickle
import os
from sys import argv

sys.path.append("..")
from transformers.StaticTransformer import StaticTransformer
from transformers.LastStateTransformer import LastStateTransformer
from transformers.AggregateTransformer import AggregateTransformer
from transformers.IndexBasedTransformer import IndexBasedTransformer
from transformers.HMMDiscriminativeTransformer import HMMDiscriminativeTransformer
from transformers.HMMGenerativeTransformer import HMMGenerativeTransformer

import dataset_confs

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

home_dir = ".."
results_dir = "final_results3"
optimal_params_filename = "optimal_params3.pickle"

if not os.path.exists(os.path.join(home_dir, results_dir)):
    os.makedirs(os.path.join(home_dir, results_dir))

dataset_ref = argv[1]
method_name = argv[2]

with open(os.path.join(home_dir, optimal_params_filename), "rb") as fin:
    best_params = pickle.load(fin)

dataset_ref_to_datasets = {
    "bpic2011": ["bpic2011_f%s"%formula for formula in range(1,5)],
    #"bpic2015_f1": ["bpic2015_%s_f1"%(municipality) for municipality in range(1,6)],
    "bpic2015": ["bpic2015_%s_f2"%(municipality) for municipality in range(1,6)],
    "insurance": ["insurance_activity", "insurance_followup"],
    "sepsis_cases": ["sepsis_cases"],
    "traffic_fines": ["traffic_fines"],
    "siae": ["siae"],
    "bpic2017": ["bpic2017"]
}
dataset_ref_to_datasets["small_logs"] = dataset_ref_to_datasets["bpic2011"] + dataset_ref_to_datasets["bpic2015"] + dataset_ref_to_datasets["insurance"] + dataset_ref_to_datasets["sepsis_cases"]


methods_dict = {
    "cluster_laststate": ["static", "laststate"],
    "cluster_agg": ["static", "agg"],
    "cluster_hmm": ["static", "hmm_disc"]}
    #"cluster_combined": ["static", "laststate", "agg", "hmm_disc"]}
    
datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
methods = methods_dict[method_name]

outfile = os.path.join(home_dir, results_dir, "final_results_%s_%s.csv"%(method_name, dataset_ref)) 

    
train_ratio = 0.8
hmm_min_seq_length = 2
hmm_max_seq_length = None
hmm_n_iter = 50
hmm_n_states = None # assigned on the fly from the best params
rf_n_estimators = 500
random_state = 22
rf_max_features = None # assigned on the fly from the best params
n_clusters = None # assigned on the fly from the best params
fillna = True



def init_encoder(method):
    
    if method == "static":
        return StaticTransformer(case_id_col=case_id_col, cat_cols=static_cat_cols, num_cols=static_num_cols, fillna=fillna)
    
    elif method == "laststate":
        return LastStateTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, fillna=fillna)
    
    elif method == "agg":
        return AggregateTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, fillna=fillna)
    
    elif method == "hmm_disc":
        hmm_disc_encoder = HMMDiscriminativeTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols,
                                                        num_cols=dynamic_num_cols, n_states=hmm_n_states, label_col=label_col,
                                                        pos_label=pos_label, min_seq_length=hmm_min_seq_length,
                                                        max_seq_length=hmm_max_seq_length, random_state=random_state,
                                                        n_iter=hmm_n_iter, fillna=fillna)
        return hmm_disc_encoder
    
    elif method == "hmm_gen":
        hmm_gen_encoder = HMMGenerativeTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols,
                                                   num_cols=dynamic_num_cols, n_states=hmm_n_states,
                                                   min_seq_length=hmm_min_seq_length, max_seq_length=hmm_max_seq_length,
                                                   random_state=random_state, n_iter=hmm_n_iter, fillna=fillna)
        return hmm_gen_encoder
    
    else:
        print("Invalid encoder type")
        return None
    
    
##### MAIN PART ######    
with open(outfile, 'w') as fout:
    
    fout.write("%s;%s;%s;%s;%s\n"%("dataset", "method", "nr_events", "metric", "score"))
    
    for dataset_name in datasets:
        
        # read dataset settings
        case_id_col = dataset_confs.case_id_col[dataset_name]
        activity_col = dataset_confs.activity_col[dataset_name]
        timestamp_col = dataset_confs.timestamp_col[dataset_name]
        label_col = dataset_confs.label_col[dataset_name]
        pos_label = dataset_confs.pos_label[dataset_name]

        dynamic_cat_cols = dataset_confs.dynamic_cat_cols[dataset_name]
        static_cat_cols = dataset_confs.static_cat_cols[dataset_name]
        dynamic_num_cols = dataset_confs.dynamic_num_cols[dataset_name]
        static_num_cols = dataset_confs.static_num_cols[dataset_name]
        
        data_filepath = os.path.join(home_dir, dataset_confs.filename[dataset_name])

        dtypes = {col:"object" for col in dynamic_cat_cols+static_cat_cols+[case_id_col, label_col, timestamp_col]}
        for col in dynamic_num_cols + static_num_cols:
            dtypes[col] = "float"
        
        data = pd.read_csv(data_filepath, sep=";", dtype=dtypes)
        data[timestamp_col] = pd.to_datetime(data[timestamp_col])
        
        # consider prefix lengths until 90% of positive cases have finished
        min_prefix_length = 1 if "hmm_disc" not in methods else 2
        prefix_lengths = list(range(min_prefix_length, min(20, int(np.ceil(data.groupby(case_id_col).size().quantile(0.75)))) + 1))
        #prefix_lengths = list(range(min_prefix_length, min(20, int(np.ceil(data[data[label_col]==pos_label].groupby(case_id_col).size().quantile(0.90)))) + 1))

        # split into train and test using temporal split
        grouped = data.groupby(case_id_col)
        start_timestamps = grouped[timestamp_col].min().reset_index()
        start_timestamps.sort_values(timestamp_col, ascending=1, inplace=True)
        train_ids = list(start_timestamps[case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[case_id_col].isin(train_ids)].sort_values(timestamp_col, ascending=True)
        test = data[~data[case_id_col].isin(train_ids)].sort_values(timestamp_col, ascending=True)
        del data
        del start_timestamps
        del train_ids
        
        train['case_length'] = train.groupby(case_id_col)[activity_col].transform(len)

        test_case_lengths = test.groupby(case_id_col).size()

        # generate prefix data (each possible prefix becomes a trace)
        print("Generating prefix data...")
        sys.stdout.flush()
        train_prefixes = train[train['case_length'] >= prefix_lengths[0]].groupby(case_id_col).head(prefix_lengths[0])
        for nr_events in prefix_lengths[1:]:
            tmp = train[train['case_length'] >= nr_events].groupby(case_id_col).head(nr_events)
            tmp[case_id_col] = tmp[case_id_col].apply(lambda x: "%s_%s"%(x, nr_events))
            train_prefixes = pd.concat([train_prefixes, tmp], axis=0)
        
        if dataset_name not in best_params or method_name not in best_params[dataset_name]:
            continue
        hmm_n_states = best_params[dataset_name][method_name]['hmm_n_states']
        rf_max_features = best_params[dataset_name][method_name]['rf_max_features']
        n_clusters = best_params[dataset_name][method_name]['n_clusters']

        # cluster prefixes based on control flow
        print("Clustering prefixes...")
        start = time()
        freq_encoder = AggregateTransformer(case_id_col=case_id_col, cat_cols=[activity_col], num_cols=[], fillna=fillna)
        data_freqs = freq_encoder.fit_transform(train_prefixes)
        clustering = KMeans(n_clusters, random_state=random_state)
        cluster_assignments = clustering.fit_predict(data_freqs)
        clustering_time = time() - start
            
        pipelines = {}

        start = time()
        # train and fit pipeline for each cluster
        for cl in range(n_clusters):
            print("Fitting pipeline for cluster %s..."%cl)
            relevant_cases = data_freqs[cluster_assignments == cl].index

            if len(relevant_cases) == 0:
                continue
            
            cls = RandomForestClassifier(n_estimators=rf_n_estimators, max_features=rf_max_features, random_state=random_state)
            feature_combiner = FeatureUnion([(method, init_encoder(method)) for method in methods])
            pipelines[cl] = Pipeline([('encoder', feature_combiner), ('cls', cls)])

            # fit pipeline
            dt_train_cluster = train_prefixes[train_prefixes[case_id_col].isin(relevant_cases)].sort_values(timestamp_col, ascending=True)
            train_y = dt_train_cluster.groupby(case_id_col).first()[label_col]
            pipelines[cl].fit(dt_train_cluster, train_y)
        total_encoding_cls_time = time() - start

        fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, 0, "clustering_time", clustering_time))
        fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, 0, "total_encoding_cls_time", total_encoding_cls_time))
        
        # test separately for each prefix length
        for nr_events in prefix_lengths:
            print("Predicting for %s events..."%nr_events)

            # select only cases that are at least of length nr_events
            relevant_case_ids = test_case_lengths.index[test_case_lengths >= nr_events]
            if len(relevant_case_ids) == 0:
                break
            relevant_test = test[test[case_id_col].isin(relevant_case_ids)].sort_values(timestamp_col, ascending=True)
                
            del relevant_case_ids
                    
            start = time()
            # get predicted cluster for each test case
            test_data_freqs = freq_encoder.transform(relevant_test.groupby(case_id_col).head(nr_events))
            test_cluster_assignments = clustering.predict(test_data_freqs)

            # use appropriate classifier for each bucket of test cases
            preds = []
            test_y = []
            for cl in range(n_clusters):
                current_cluster_case_ids = test_data_freqs[test_cluster_assignments == cl].index
                current_cluster_grouped_test = relevant_test[relevant_test[case_id_col].isin(current_cluster_case_ids)].sort_values(timestamp_col, ascending=True).groupby(case_id_col, as_index=False)
                    
                if len(current_cluster_case_ids) == 0:
                    continue
                elif cl not in pipelines:
                    # no classifier exists for this cluster, hardcode predictions
                    current_cluster_preds = [0.5] * len(current_cluster_case_ids)
                elif len(pipelines[cl].named_steps["cls"].classes_) == 1:
                    hardcoded_prediction = 1 if pipelines[cl].named_steps["cls"].classes_[0] == pos_label else 0
                    current_cluster_preds = [hardcoded_prediction] * len(current_cluster_case_ids)
                else:
                    # make predictions
                    preds_pos_label_idx = np.where(pipelines[cl].named_steps["cls"].classes_ == pos_label)[0][0] 
                    current_cluster_preds = pipelines[cl].predict_proba(current_cluster_grouped_test.head(nr_events))[:,preds_pos_label_idx]
                        
                preds.extend(current_cluster_preds)

                # extract actual label values
                current_cluster_test_y = [1 if label==pos_label else 0 for label in current_cluster_grouped_test.first()[label_col]]
                test_y.extend(current_cluster_test_y)
            total_prediction_time = time() - start

            if len(set(test_y)) < 2:
                auc = None
            else:
                auc = roc_auc_score(test_y, preds)
            prec, rec, fscore, _ = precision_recall_fscore_support(test_y, [0 if pred < 0.5 else 1 for pred in preds], average="binary")

            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "auc", auc))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "precision", prec))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "recall", rec))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "fscore", fscore))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "total_prediction_time", total_prediction_time))
                
        print("\n")
