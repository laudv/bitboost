import string
import numpy as np
import pandas as pd
from datetime import datetime

def rand_filename(prefix):
    now = datetime.now()
    alphabet = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
    rand_suffix = ''.join(np.random.choice(alphabet, 3))
    result_filename = '{}-{}-{}'.format(prefix, now.strftime('%y%m%d-%H%M%S'), rand_suffix)
    return result_filename

def average_results(df):
    grouped = df.groupby("hash")
    df_grouped = pd.concat([
        grouped[['method', 'objective', 'learning_rate', 'feature_fraction', 'example_fraction',
                 'max_depth', 'niterations', 'bit_sample_freq', 'bit_discr_nbits', 'bit_compr_threshold',
                 'bit_max_nbins']].agg(["first"]),
        grouped[['train_time', 'metric_train', 'metric_val', 'metric_test']].agg(["mean", "std"]),
        grouped["hash"].count().to_frame("count")
    ], axis=1).sort_values(("method", "first"))
    return df_grouped