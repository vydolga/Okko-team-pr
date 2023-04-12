import pandas as pd
from typing import Dict

def prepare_lfm_for_train(paths_config: Dict[str, str]):
    import datetime as dt
    from utils.utils import read_csv_from_gdrive

    INTERACTIONS_PATH = paths_config.get('INTERACTIONS_PATH')
    interactions = read_csv_from_gdrive(INTERACTIONS_PATH)

    # remove redundant data points
    interactions_filtered = interactions.loc[interactions['total_dur'] > 300].reset_index(drop=True)
    interactions_filtered['last_watch_dt'] = pd.to_datetime(interactions_filtered['last_watch_dt'])

    # set dates params for filter
    MAX_DATE = interactions_filtered['last_watch_dt'].max()
    MIN_DATE = interactions_filtered['last_watch_dt'].min()
    TEST_INTERVAL_DAYS = 14

    TEST_MAX_DATE = MAX_DATE - dt.timedelta(days=TEST_INTERVAL_DAYS)

    # define global train and test
    global_train = interactions_filtered.loc[interactions_filtered['last_watch_dt'] < TEST_MAX_DATE]
    global_test = interactions_filtered.loc[interactions_filtered['last_watch_dt'] >= TEST_MAX_DATE]

    # now, we define "local" train and test to use some part of the global train for ranker
    local_train_thresh = global_train['last_watch_dt'].quantile(q=.7, interpolation='nearest')

    global_train = global_train.dropna().reset_index(drop=True)

    local_train = global_train.loc[global_train['last_watch_dt'] < local_train_thresh]
    local_test = global_train.loc[global_train['last_watch_dt'] >= local_train_thresh]

    # finally, we will focus on warm start -- remove cold start users
    local_test = local_test.loc[local_test['user_id'].isin(local_train['user_id'].unique())]

    return local_train, local_test