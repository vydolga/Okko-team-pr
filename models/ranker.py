# Here, you run both models training pipeline using modules we created
# LFM - load wathch data and run fit() method
# Ranker - load candidates based data with features and run fit() method
# REMINDER: it must be active and working. Before that, you shoul finalize prepare_ranker_data.py
import logging
from typing import Dict, List
# import catboost as cb
from catboost import CatBoostClassifier
import pandas as pd
import os


class Ranker:
    def __init__(self, model_path: str = '/Users/vydolga/Documents/HSE MA 24/Okko project/artefacts/catboost_clf.cbm'):
        logging.info("loading the model")
        self.ranker = CatBoostClassifier().load_model(fname=model_path)
        
    @staticmethod
    def fit(
            X_train: pd.DataFrame,
            y_train: pd.DataFrame,
            X_test: pd.DataFrame = None,
            y_test: pd.DataFrame = None,
            ranker_params: dict = None,
            categorical_cols: list = None,
        ) -> None:
        
        cbm_classifier = CatBoostClassifier(
        loss_function = 'CrossEntropy',
        iterations = 5000,
        learning_rate = .1,
        depth = 6,
        random_state = 1234,
        verbose = True
        )

        logging.info("fit ranker.py")
        CATEGORICAL_COLS=['age', 'income', 'sex', 'content_type']
        cbm_classifier.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds = 100, # to avoid overfitting,
        cat_features = CATEGORICAL_COLS)

        logging.info("Saving the model ranker.py")
        current_directory = os.path.dirname(os.path.realpath(__file__))
        cbm_classifier.save_model( f'{current_directory}/artefacts/catboost_clf.cbm')
    
    def infer(self, ranker_input: List) -> Dict[str, int]:
        logging.info("Making predictions ranker.py")
        preds = self.ranker.predict_proba(ranker_input)[:, 1]

        return preds
    
