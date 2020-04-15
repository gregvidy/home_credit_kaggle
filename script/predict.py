import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
import numpy as np
from .scorer import Scorer
import math

from . import dispatcher

TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")

def extract_score(prob):
    score = Scorer(min_score=300,
                   max_score=850,
                   pdo=30,
                   base_odds=math.exp(3),
                   base=600)
    return score.to_score(prob[1])

def predict():
    df = pd.read_csv(TEST_DATA)
    test_idx = df["LN_ID"].values
    prob_preds = None
    score_preds = None

    for FOLD in range(5):
        print(FOLD)
        df = pd.read_csv(TEST_DATA)
        cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))        
           
        # predict the data
        clf = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}.pkl"))
        
        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]
        scores = list(map(lambda x: extract_score(x), clf.predict_proba(df)))

        if FOLD == 0:
            prob_preds = preds
            score_preds = scores
        else:
            prob_preds += preds
            score_preds += scores
     
    prob_preds /= 5
    score_preds /= 5 #= [x / 5 for x in score_preds]
    label_preds = [1 if x >= 0.5 else 0 for x in prob_preds]

    sub = pd.DataFrame(np.column_stack((test_idx, prob_preds, label_preds, score_preds)),
                                        columns=["LN_ID", "PROB_PREDS", "LABEL_PREDS", "SCORE_PREDS"])
    sub["LN_ID"] = sub["LN_ID"].astype(int)
    return sub

if __name__ == "__main__":
    print("==============================")
    print("Start predicting ...")
    submission = predict()
    submission.to_csv(f"models/{MODEL}.csv", index=False)
    print("Prediction saved!")
