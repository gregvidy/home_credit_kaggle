import pandas as pd
import numpy as np
import pickle


def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == "object"]
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def application_train_test(num_rows=None, nan_as_categegory=True):
    app_train = pd.read_csv("../input/app_train.csv", nrows=num_rows, sep=",", index_col=0)
    app_test = pd.read_csv("../input/app_test.csv", nrows=num_rows, sep=",", index_col=0)

    # store y_actual then replacing app_test target with nan
    y_actual = app_test["TARGET"].copy()
    app_test["TARGET"].replace(0, np.nan, inplace=True)
    app_test["TARGET"].replace(1, np.nan, inplace=True)

    # join the data
    app_train_test = pd.concat([app_train, app_test], axis=0)

    # replace anomalous value
    app_train_test["DAYS_WORK"].replace(365243, np.nan, inplace=True)

    # create new features
    income_by_org = app_train_test[["INCOME", "ORGANIZATION_TYPE"]].groupby("ORGANIZATION_TYPE").median()["INCOME"]
    app_train_test["NEW_ANNUITY_TO_CREDIT_RATIO"] = app_train_test["ANNUITY"] / app_train_test["APPROVED_CREDIT"]
    app_train_test["NEW_CREDIT_TO_GOODS_RATIO"] = app_train_test["APPROVED_CREDIT"] / app_train_test["PRICE"]
    app_train_test["NEW_INCOME_BY_ORG"] = app_train_test["ORGANIZATION_TYPE"].map(income_by_org)
    app_train_test["NEW_EMPLOYED_TO_BIRTH_RATIO"] = app_train_test["DAYS_WORK"] / app_train_test["DAYS_AGE"]
    app_train_test["NEW_ANNUITY_TO_INCOME_RATIO"] = app_train_test["ANNUITY"] / app_train_test["INCOME"]
    app_train_test["NEW_CREDIT_TO_INCOME_RATIO"] = app_train_test["APPROVED_CREDIT"] / app_train_test["INCOME"]
    app_train_test["NEW_GOODS_TO_INCOME_RATIO"] = app_train_test["PRICE"] / app_train_test["INCOME"]
    app_train_test["NEW_SCORE_PROD"] = app_train_test["EXT_SCORE_1"] * app_train_test["EXT_SCORE_2"] * app_train_test["EXT_SCORE_3"]
    app_train_test["NEW_EXT_SCORES_MEAN"] = app_train_test[["EXT_SCORE_1", "EXT_SCORE_2", "EXT_SCORE_3"]].mean(axis=1)
    app_train_test["NEW_EXT_SCORES_STD"] = app_train_test[["EXT_SCORE_1", "EXT_SCORE_2", "EXT_SCORE_3"]].std(axis=1)
    app_train_test["NEW_EXT_SCORES_STD"] = app_train_test["NEW_EXT_SCORES_STD"].fillna(app_train_test["NEW_EXT_SCORES_STD"].mean())
    app_train_test['NEW_INC_PER_PERSON'] = app_train_test['INCOME'] / (1 + app_train_test['NUM_CHILDREN'])

    # encoding categorical columns
    app_train_test, cat_cols = one_hot_encoder(app_train_test, nan_as_category=True)

    app_train_test.set_index("LN_ID")
    del app_train, app_test, cat_cols
    return app_train_test, y_actual


def previous_applications(num_rows=None, nan_as_category=True):
    prev_app = pd.read_csv("DS1/prev_app.csv", nrows=num_rows, sep=",", index_col=0)
    
    # replace anomalous value
    prev_app["FIRST_DRAW"].replace(365243, np.nan, inplace=True)
    prev_app["FIRST_DUE"].replace(365243, np.nan, inplace=True)
    prev_app["TERMINATION"].replace(365243, np.nan, inplace=True)
    
    # new feature
    prev_app["CREDIT_PERC"] = prev_app["APPLICATION"] / prev_app["APPROVED_CREDIT"]
    
    # aggregation
    num_aggregations = {"ANNUITY": ["max", "mean"],
                        "APPLICATION": ["min", "mean"],
                        "CREDIT_PERC": ["min", "max", "mean"],
                        "APPROVED_CREDIT": ["min", "max", "mean"],
                        "AMT_DOWN_PAYMENT": ["min", "max", "mean"],
                        "PRICE": ["min", "max", "mean"],
                        "HOUR_APPLY": ["min", "max", "mean"],
                        "DAYS_DECISION": ["min", "max", "mean"],
                        "TERM_PAYMENT": ["mean", "sum"],
                        "FIRST_DRAW": ["min", "max", "mean"],
                        "FIRST_DUE": ["min", "max", "mean"],
                        "TERMINATION": ["min", "max", "mean"]}
    
    # encoding categorical
    prev_app["NFLAG_INSURED_ON_APPROVAL"] = prev_app["NFLAG_INSURED_ON_APPROVAL"].astype("object")
    prev_app, cat_cols = one_hot_encoder(prev_app, nan_as_category=True)
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ["mean"]
    
    # group by aggregation
    prev_agg = prev_app.groupby("LN_ID").agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(["PREV_" + c[0] + "_" + c[1].upper() for c in prev_agg.columns.tolist()])
    
    # group by previous application: approved - only numerical features
    approved = prev_app[prev_app["CONTRACT_STATUS_Approved"] == 1]
    approved_agg = approved.groupby("LN_ID").agg(num_aggregations)
    approved_agg.columns = pd.Index(["APPROVED_" + c[0] + "_" + c[1].upper() for c in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how="left", on="LN_ID")
    
    # group by previous application: refused - only numerical features
    refused = prev_app[prev_app["CONTRACT_STATUS_Refused"] == 1]
    refused_agg = refused.groupby("LN_ID").agg(num_aggregations)
    refused_agg.columns = pd.Index(["REFUSED_" + c[0] + "_" + c[1].upper() for c in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how="left", on="LN_ID")
    
    del refused, refused_agg, approved, approved_agg, prev_app
    return prev_agg


def installment_payment(num_rows=None, nan_as_category=True):
    installment_pmt = pd.read_csv("DS1/installment_payment.csv", nrows=num_rows, sep=",", index_col=0)

    # create new features
    installment_pmt["PAYMENT_PERC"] = installment_pmt["AMT_PAY"] / installment_pmt["AMT_INST"]
    installment_pmt["PAYMENT_DIFF"] = installment_pmt["AMT_PAY"] - installment_pmt["AMT_INST"]
    installment_pmt["OVERDUE_DAYS"] = installment_pmt["PAY_DAYS"] - installment_pmt["INST_DAYS"]
    has_due = installment_pmt["OVERDUE_DAYS"] > 0
    hasnt_due = installment_pmt["OVERDUE_DAYS"] <= 0
    installment_pmt.loc[has_due, "HAS_DUE"] = 1
    installment_pmt.loc[hasnt_due, "HAS_DUE"] = 0

    # aggregations
    aggregations = {"INST_NUMBER": ["nunique"],
                    "INST_DAYS": ["max", "mean", "sum"],
                    "PAY_DAYS": ["max", "mean", "sum"],
                    "AMT_INST": ["max", "mean", "sum"],
                    "AMT_PAY": ["min", "max", "mean", "sum"],
                    "PAYMENT_PERC": ["mean", "sum", "var"],
                    "PAYMENT_DIFF": ["mean", "sum", "var"],
                    "OVERDUE_DAYS": ["min", "max", "mean"],
                    "HAS_DUE": ["max", "mean", "sum"]}
    
    # encoding categorical variables
    installment_pmt, cat_cols = one_hot_encoder(installment_pmt, nan_as_category=True)
    for cat in cat_cols:
        aggregations[cat] = ["mean"]
    
    # group by
    installment_agg = installment_pmt.groupby("LN_ID").agg(aggregations)
    installment_agg.columns = pd.Index(['INSTAL_' + c[0] + "_" + c[1].upper() for c in installment_agg.columns.tolist()])
    installment_agg["INSTAL_COUNT"] = installment_pmt.groupby('LN_ID').size()

    installment_pmt.set_index("LN_ID")
    del installment_pmt
    return installment_agg

def main(num_rows):
    print('Reading data into memory ...')
    df, y_actual = application_train_test(num_rows)
    prev = previous_applications(num_rows)
    installment = installment_payment(num_rows)

    print("Previous applications df shape:", prev.shape)
    df = df.join(prev, how="left", on="LN_ID")
    del prev

    print("Installment amount df shape:", installment.shape)
    df = df.join(installment, how="left", on="LN_ID")
    del installment

    print("Final df shape:", df.shape)
    print('Preprocessing is complete, saving data to disk ...')
    return pickle.dump([df, y_actual], open("preprocessed_data", "wb")), print("Data saved!")

if __name__ == "__main__":
    main(2872306)