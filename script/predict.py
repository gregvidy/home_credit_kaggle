import math
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt     
from scorer import Scorer
from score_utils import plot
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# load data and label
df, y_actual = pickle.load(open("preprocessed_data", "rb"))
test_df = df[df["TARGET"].isnull()]
feats = [f for f in df.columns if f not in ["LN_ID", "TARGET"]]

#load model 
model_xgbt = pickle.load(open("model_xgbt_v1", "rb"))

# classification report to dataframe
def export_clf_report(to_csv=False):
    report = classification_report(y_actual, model_xgbt.predict(test_df[feats]), output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    return df_report.to_csv('classification_report.csv', sep=';', index=True)

# confusion matrix plot
def confusion_matrix_plot(save_plot=False):
    plt.figure(figsize=(6,5))
    ax = plt.subplot()
    cm = confusion_matrix(y_actual, model_xgbt.predict(test_df[feats]))
    sns.heatmap(cm, annot=True, ax = ax, cmap="Blues", fmt="g") # annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    if save_plot:
        return plt.savefig('plot_confusion_matrix.png')

# extract score
def extract_score(prob):
    score = Scorer(min_score=300, max_score=850, pdo=30, base_odds=math.exp(3), base=600)
    return score.to_score(prob[1])

def score_band_plot():
    tmp = list(map(lambda x: extract_score(x),model_xgbt.predict_proba(test_df[feats])))
    t1 = plot.plot_triple(pd.Series(tmp),y_actual>0,10,table=True)
    return t1

# extract score table
def extract_score_band():
    t2 = score_band_plot()
    return t2.to_csv('score_band.csv', sep=';')

def main():
    print("Export classification report ...")
    export_clf_report(to_csv=True)
    print("Export confusion matrix plot ...")
    confusion_matrix_plot(save_plot=True)
    print("Export score band plot ...")
    score_band_plot()
    print("Export score band table ...")
    extract_score_band()
    print("Export finish!")

if __name__ == "__main__":
    main()