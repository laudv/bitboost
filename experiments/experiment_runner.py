import sys, io, os
import numpy as np
import pandas as pd
import multiprocessing as mp
import datetime

from functools import partial
from sklearn.model_selection import KFold, train_test_split
from gbdt_model import XGBModel, LGBModel, CatModel, BitModel, Dataset
from params import params_to_csv_row, params_to_csv_header


def get_model(model_name):
    if model_name == "xgb":
        return XGBModel()
    elif model_name == "lgb":
        return LGBModel()
    elif model_name == "cat":
        return CatModel()
    elif model_name == "bit":
        return BitModel()
    else:
        raise Exception("unknown model {}".format(model_name))

def run_experiment_for_params(df, params, random_state=1, nfolds=5):
    out = io.StringIO()

    kf = KFold(n_splits=nfolds, random_state=random_state, shuffle=True)
    for (fold_i, (train_val_indexes, test_indexes)) in enumerate(kf.split(df)):
        now = datetime.datetime.now()
        h = now.hour
        m = now.minute
        s = now.second
        print("[pid={}] {:02}:{:02}:{:02} {} fold {}/{}".format(os.getpid(), h, m, s,
                                                                params["method"],
                                                                fold_i+1, nfolds))
        df_train_val = df.iloc[train_val_indexes]
        df_train, df_val = train_test_split(df_train_val,
            test_size=len(test_indexes), random_state=random_state+fold_i)
        df_test = df.iloc[test_indexes]

        target = params["target"]
        model = get_model(params["method"])

        # Validate set performance for param selection (afterwards)
        model.set_params(params)
        model.set_data(Dataset(df_train, target), Dataset(df_val, target))
        train_time0, metric_train0, metric_val = model.train()

        # Test set performance for final quality measure
        model.set_params(params)
        model.set_data(Dataset(df_train_val, target), Dataset(df_test, target))
        train_time, metric_train, metric_test = model.train()

        record_fields = {
            "train_time0":   train_time0,
            "train_time":    train_time,
            "metric_train0": metric_train0,
            "metric_train":  metric_train,
            "metric_val":    metric_val,
            "metric_test":   metric_test,
            "fold":          fold_i
        }

        print(params_to_csv_row(params, record_fields), file=out)

    now = datetime.datetime.now()
    h = now.hour
    m = now.minute
    s = now.second
    print("[pid={}] {:02}:{:02}:{:02} {} done".format(os.getpid(), h, m, s, params["method"]))

    out.seek(0)
    return out.read()

def run_all(df, params_list, random_state=1, nfolds=5, nprocs=1, outfile=None):
    chunksize = min(20, max(1, int(len(params_list) / nprocs)))

    with mp.Pool(nprocs) as pool:
        f = partial(run_experiment_for_params, df, random_state=random_state,
                nfolds=nfolds)
        csv_rows = "".join(pool.map(f, params_list, chunksize=chunksize))

    csv = io.StringIO()
    print(params_to_csv_header(), file=csv)
    print(csv_rows, file=csv)
    csv.seek(0)

    df = pd.read_csv(csv)

    if outfile is not None:
        df.to_csv(outfile, index=False)

    return df

