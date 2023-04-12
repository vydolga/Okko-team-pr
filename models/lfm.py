import logging
from typing import Any, Dict
import dill
import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from tqdm import tqdm


class LFMModel:
    def __init__(self):
        pass

   
    def df_to_tuple_iterator(df: pd.DataFrame):
        '''
        :df: pd.DataFrame, interactions dataframe
        returs iterator
        '''
        return zip(*df.values.T)

    def concat_last_to_list(t):
        return (t[0], list(t[1:])[0])

    
    def fit(
        self,
        data: pd.DataFrame,
        user_col: str,
        item_col: str,
        model_params: Dict[str, Any] = {},
    ) -> None:

        # init class
        dataset = Dataset()

        logging.info("fit tuple of user and movie interactions")
        dataset.fit(data['user_id'].unique(), data['item_id'].unique())

        logging.info("defining train set on the whole interactions dataset (as HW you will have to split into test and train for evaluation)")
        train_mat, train_mat_weights = dataset.build_interactions(self.df_to_tuple_iterator(data[['user_id', 'item_id']]))

        logging.info("save mappers")
        with open(f"./artefacts/lfm_mapper.dill", "wb") as mapper_file:
            dill.dump(dataset, mapper_file)

        NO_COMPONENTS = 64
        LEARNING_RATE = .03
        LOSS = 'warp'
        MAX_SAMPLED = 5
        RANDOM_STATE = 42
        EPOCHS = 20

        lfm_model = LightFM(
            no_components = NO_COMPONENTS,
            learning_rate = LEARNING_RATE,
            loss = LOSS,
            max_sampled = MAX_SAMPLED,
            random_state = RANDOM_STATE
            )
        
        logging.info("execute training")
        for _ in tqdm(range(EPOCHS), total = EPOCHS):
            lfm_model.fit_partial(
                train_mat,
                num_threads = 4
            )
        logging.info("save model")
        with open(f"./artefacts/lfm_model.dill", "wb") as model_file:
            dill.dump(lfm_model, model_file)
    
    def infer(user_id: int, top_N: int = 20) -> Dict[str, int]:

        with open(f"./artefacts/lfm_model.dill", "rb") as model_file:
            lfm_model = dill.load(model_file)

        with open(f"./artefacts/lfm_mapper.dill", "rb") as mapper_file:
            dataset = dill.load(mapper_file)

        lightfm_mapping = dataset.mapping()

        logging.info("here we create inverted mappers to check recommendations later")
        users_inv_mapping = {v: k for k, v in lightfm_mapping['users_mapping'].items()}
        items_inv_mapping = {v: k for k, v in lightfm_mapping['items_mapping'].items()}


        row_id = lightfm_mapping['users_mapping'][user_id]
        all_cols = list(lightfm_mapping['items_mapping'].values())

        pred = lfm_model.predict(
            row_id,
            all_cols,
            num_threads = 4)
        
        top_cols = np.argpartition(pred, -np.arange(top_N))[-top_N:][::-1]
        item_pred_list = []
        for item in top_cols:
            item_pred_list.append(items_inv_mapping[item])
            final_preds = {v: k + 1 for k, v in enumerate(item_pred_list)}

        return final_preds


