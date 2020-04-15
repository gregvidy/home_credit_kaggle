import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
import numpy as np

from . import dispatcher

TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")

def predict():
    df = pd.read_csv(TEST_DATA)
    test_idx = df["PassengerId"].values
    predictions = None

    for FOLD in range(5):
        print(FOLD)
        df = pd.read_csv(TEST_DATA)
        cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))        
           
        # predict the data
        clf = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}.pkl"))
        
        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds
    
    predictions /= 5
    label_predictions = [1 if x >= 0.5 else 0 for x in predictions]
    
    sub = pd.DataFrame(np.column_stack((test_idx, label_predictions)), columns=["PassengerId", "TARGET"])
    sub["PassengerId"] = sub["PassengerId"].astype(int)
    return sub

if __name__ == "__main__":
    submission = predict()
    submission.to_csv(f"models/{MODEL}.csv", index=False)
