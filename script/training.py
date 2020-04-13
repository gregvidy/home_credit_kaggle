import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import KFold, StratifiedKFold
import pickle

def kfold_xgb(num_folds, stratified=False): # running model and store to pickle file
    print("Training model start ...")
    df, y_actual = pickle.load(open("preprocessed_data", "rb"))
    
    # divide in training/validation and test data
    train_df = df[df["TARGET"].notnull()]
    print("Train df shape:", train_df.shape)
    test_df = df[df["TARGET"].isnull()]
    print("Test df shape:", test_df.shape)
    del df
    
    # cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=123)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=123)
    
    # create arrays and dataframe to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feats = [f for f in train_df.columns if f not in ["LN_ID", "TARGET"]]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df["TARGET"])):
        if n_fold == 0: # remove for full K-Fold run
            train_x, train_y = train_df[feats].iloc[train_idx], train_df["TARGET"].iloc[train_idx]
            valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df["TARGET"].iloc[valid_idx]
            
            # set hyperparameter
            clf = XGBClassifier(max_depth=5,
                                min_child_weight=2,
                                subsample=0.8,
                                colsample_bytree=0.75,
                                gamma=0.2,
                                scale_pos_weight=11.3,
                                reg_alpha=100,
                                n_estimator=2000,
                                seed=1997)
            clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], eval_metric="auc", verbose=100, early_stopping_rounds=200)
            oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]
            sub_preds += clf.predict_proba(test_df[feats])[:, 1]
            
            print("Fold %2d AUC: %.6f" % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
            del train_x, train_y, valid_x, valid_y
        
        np.save("xgb_oof_preds_1", oof_preds)
        np.save("xgb_sub_pred_1", sub_preds)
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_actual, clf.predict_proba(test_df[feats])[:, 1])
    print("AUC for y_actual: %.6f" % auc(false_positive_rate, true_positive_rate))
    
    del train_df, test_df, thresholds
    print("Training complete, saving model to pickle file ...")
    filename = 'model_xgbt_v1'
    
    return pickle.dump(clf, open(filename, 'wb')), print("Model saved!")

if __name__ == "__main__":
    kfold_xgb(15, stratified=True)