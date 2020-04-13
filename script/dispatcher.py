from sklearn import ensemble
from sklearn import linear_model
import xgboost

MODELS = {
    "xgboost": xgboost.XGBClassifier(max_depth=5,
                                     min_child_weight=2,
                                     subsample=0.8,
                                     colsample_bytree=0.75,
                                     gamma=0.2,
                                     scale_pos_weight=11.3, # change with the ratio between good/bad distribution sample
                                     reg_alpha=100,
                                     n_estimator=2000,
                                     seed=1997,
                                     verbose=100,
                                     early_stopping_rounds=200),

    "randomforest": ensemble.RandomForestClassifier(n_estimators=100,
                                                    n_jobs=-1,
                                                    verbose=2,
                                                    random_state=1997),

    "extratrees": ensemble.RandomForestClassifier(n_estimators=200,
                                                  n_jobs=-1,
                                                  verbose=2,
                                                  random_state=123),

    "logreg": linear_model.LogisticRegression(penalty="l2",
                                              C=0.0001,
                                              max_iter=150,
                                              verbose=2,
                                              n_jobs=-1,
                                              class_weight="balanced") # change if the percentage between two classes is balanced/not balanced
}