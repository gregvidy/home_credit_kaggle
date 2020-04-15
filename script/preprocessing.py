import pandas as pd
import numpy as np
import pickle
from categorical_features import CategoricalFeatures

def application_train_test(df):
    # replace anomalous value
    df["DAYS_WORK"].replace(365243, np.nan, inplace=True)

    # create new features
    income_by_org = df[["INCOME", "ORGANIZATION_TYPE"]].groupby("ORGANIZATION_TYPE").median()["INCOME"]
    df["NEW_ANNUITY_TO_CREDIT_RATIO"] = df["ANNUITY"] / df["APPROVED_CREDIT"]
    df["NEW_CREDIT_TO_GOODS_RATIO"] = df["APPROVED_CREDIT"] / df["PRICE"]
    df["NEW_INCOME_BY_ORG"] = df["ORGANIZATION_TYPE"].map(income_by_org)
    df["NEW_EMPLOYED_TO_BIRTH_RATIO"] = df["DAYS_WORK"] / df["DAYS_AGE"]
    df["NEW_ANNUITY_TO_INCOME_RATIO"] = df["ANNUITY"] / df["INCOME"]
    df["NEW_CREDIT_TO_INCOME_RATIO"] = df["APPROVED_CREDIT"] / df["INCOME"]
    df["NEW_GOODS_TO_INCOME_RATIO"] = df["PRICE"] / df["INCOME"]
    df["NEW_SCORE_PROD"] = df["EXT_SCORE_1"] * df["EXT_SCORE_2"] * df["EXT_SCORE_3"]
    df["NEW_EXT_SCORES_MEAN"] = df[["EXT_SCORE_1", "EXT_SCORE_2", "EXT_SCORE_3"]].mean(axis=1)
    df["NEW_EXT_SCORES_STD"] = df[["EXT_SCORE_1", "EXT_SCORE_2", "EXT_SCORE_3"]].std(axis=1)
    df["NEW_EXT_SCORES_STD"] = df["NEW_EXT_SCORES_STD"].fillna(df["NEW_EXT_SCORES_STD"].mean())
    df['NEW_INC_PER_PERSON'] = df['INCOME'] / (1 + df['NUM_CHILDREN'])

    # encoding categorical columns
    cat_cols = [col for col in df.columns if df[col].dtype == "object"]
    cat_feats = CategoricalFeatures(df,
                                    categorical_features=cat_cols,
                                    encoding_type="one_hot",
                                    handle_na=True)
    df_transformed = cat_feats.fit_transform()

    df_transformed.set_index("LN_ID")
    return df_transformed


def previous_applications(df):
    # replace anomalous value
    df["FIRST_DRAW"].replace(365243, np.nan, inplace=True)
    df["FIRST_DUE"].replace(365243, np.nan, inplace=True)
    df["TERMINATION"].replace(365243, np.nan, inplace=True)
    
    # new feature
    df["CREDIT_PERC"] = df["APPLICATION"] / df["APPROVED_CREDIT"]
    
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
    df["NFLAG_INSURED_ON_APPROVAL"] = df["NFLAG_INSURED_ON_APPROVAL"].astype("object")
    cat_cols = [col for col in df.columns if df[col].dtype == "object"]
    cat_feats = CategoricalFeatures(df,
                                    categorical_features=cat_cols,
                                    encoding_type="one_hot",
                                    handle_na=True)
    df_transformed = cat_feats.fit_transform()
    df_transformed = df_transformed.loc[:,~df_transformed.columns.duplicated()] # delete duplicate columns
    new_cols = [c for c in df_transformed.columns if c not in df.columns]
    cat_aggregations = {}
    for cat in new_cols:
        cat_aggregations[cat] = ["mean"]
    
    # group by aggregation
    df_agg = df_transformed.groupby("LN_ID").agg({**num_aggregations, **cat_aggregations})
    df_agg.columns = pd.Index(["PREV_" + c[0] + "_" + c[1].upper() for c in df_agg.columns.tolist()])
    
    # group by previous application: approved - only numerical features
    approved = df_transformed[df_transformed["CONTRACT_STATUS_Approved"] == 1]
    approved_agg = approved.groupby("LN_ID").agg(num_aggregations)
    approved_agg.columns = pd.Index(["APPROVED_" + c[0] + "_" + c[1].upper() for c in approved_agg.columns.tolist()])
    df_agg = df_agg.join(approved_agg, how="left", on="LN_ID")
    
    # group by previous application: refused - only numerical features
    refused = df_transformed[df_transformed["CONTRACT_STATUS_Refused"] == 1]
    refused_agg = refused.groupby("LN_ID").agg(num_aggregations)
    refused_agg.columns = pd.Index(["REFUSED_" + c[0] + "_" + c[1].upper() for c in refused_agg.columns.tolist()])
    df_agg = df_agg.join(refused_agg, how="left", on="LN_ID")
    
    return df_agg


def installment_payment(df):
    # create new features
    df["PAYMENT_PERC"] = df["AMT_PAY"] / df["AMT_INST"]
    df["PAYMENT_DIFF"] = df["AMT_PAY"] - df["AMT_INST"]
    df["OVERDUE_DAYS"] = df["PAY_DAYS"] - df["INST_DAYS"]
    has_due = df["OVERDUE_DAYS"] > 0
    hasnt_due = df["OVERDUE_DAYS"] <= 0
    df.loc[has_due, "HAS_DUE"] = 1
    df.loc[hasnt_due, "HAS_DUE"] = 0
    
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
    cat_cols = [col for col in df.columns if df[col].dtype == "object"]
    cat_feats = CategoricalFeatures(df,
                                    categorical_features=cat_cols,
                                    encoding_type="one_hot",
                                    handle_na=True)
    df = cat_feats.fit_transform()
    for cat in cat_cols:
        aggregations[cat] = ["mean"]
    
    # group by
    df_agg = df.groupby("LN_ID").agg(aggregations)
    df_agg.columns = pd.Index(['INSTAL_' + c[0] + "_" + c[1].upper() for c in df_agg.columns.tolist()])
    df_agg["INSTAL_COUNT"] = df.groupby('LN_ID').size()

    df.set_index("LN_ID")
    return df_agg

if __name__ == "__main__":
    print('Reading data into memory ...')

    # application train test
    print('Processing application_train_test data ...')
    app_train = pd.read_csv("../input/app_train.csv")
    app_test = pd.read_csv("../input/app_test.csv")
    full_data = pd.concat([app_train, app_test], axis=0)
    app_train_test = application_train_test(full_data)
    print("Process completed!")

    # previous applications
    print('Processing previous_applications data ...')
    prev_app = pd.read_csv("../input/prev_app.csv")
    prev = previous_applications(prev_app)
    print("Process completed!")

    # installment
    print('Processing installment data ...') 
    inst_pmt = pd.read_csv("../input/installment_payment.csv")
    installment = installment_payment(inst_pmt)
    print("Process completed!")

    # joining table
    print("previous_applications df shape:", prev.shape)
    df = app_train_test.join(prev, how="left", on="LN_ID")
    del prev, app_train_test
    print("Installment amount df shape:", installment.shape)
    df = df.join(installment, how="left", on="LN_ID")
    del installment
    print("Final df shape:", df.shape)
    print('Preprocessing is complete, saving data to disk ...')
    
    # split to train_df and test_df
    train_len = len(app_train)
    df_train = df.iloc[:train_len, :]
    df_test = df.iloc[train_len:, :]
    
    # dropping columns for both train and test
    df_train.drop("LN_ID", axis=1, inplace=True)

    # saving df train and test to csv
    df_train.to_csv("../input/train_preprocessed.csv", index=False)
    print("Train data saved!")
    df_test.to_csv("../input/test_preprocessed.csv", index=False)
    print("Test data saved!") 