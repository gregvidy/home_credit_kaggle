import pandas as pd
from sklearn import preprocessing

"""
- one hot encoding
- label encoding
- binarization
"""

class CategoricalFeatures:
    def __init__(self, df, categorical_features, encoding_type, handle_na=False):
        """
        df: pandas dataframe
        categorical_features: list of categorical column names e.g. nominal, ordinal data type
        encoding_type: type of encoding e.g. label, one_hot, binary
        handle_na: handle the missing values or not e.g. True/False
        """
        self.df = df
        self.cat_feats = categorical_features
        self.enc_type  = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.one_hot_encoders = None

        if self.handle_na is True:
            for c in self.cat_feats:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-9999999")
        self.output_df = self.df.copy(deep=True)

    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:, c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df
    
    def _binarization_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.df[c].values)
            val = lbl.transform(self.df[c].values)
            self.output_df = self.output_df.drop(c, axis=1)
            for j in range(val.shape[1]):
                new_cols_name = c + f"__bin_{j}"
                self.output_df[new_cols_name] = val[:, j]
            self.binary_encoders = lbl
        return self.output_df

    def _one_hot_encoding(self):
         one_hot_encoders = preprocessing.OneHotEncoder()
         one_hot_encoders.fit(self.df[self.cat_feats].values)
         return one_hot_encoders.transform(self.df[self.cat_feats].values)

    def _get_dummies(self):
        self.output_df = pd.get_dummies(self.df, columns=self.cat_feats, dummy_na=True)
        return self.output_df

    def fit_transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        elif self.enc_type == "binary":
            return self._binarization_encoding()
        elif self.enc_type == "one_hot":
            return self._get_dummies()
        else:
            raise Exception("Encoding type not supported!")
         
    def transform(self, dataframe):
        if self.handle_na is True:
            for c in self.cat_feats:
                dataframe[:, c] = dataframe.loc[:, c].astype(str).fillna("-9999999")

        if self.enc_type == "label":
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
            return dataframe
        elif self.enc_type == "binary":
            for c, lbl in self.binary_encoders.items():
                val = lbl.transform(dataframe[c].values)
                dataframe = dataframe.drop(c, axis=1)
                for j in range(val.shape[1]):
                    new_cols_name = c + f"__bin_{j}"
                    dataframe[new_cols_name] = val[:, j]
            return dataframe
        elif self.enc_type == "one_hot":
            self.one_hot_encoders.transform(dataframe[self.cat_feats].values)
    
        else:
            raise Exception("Encoding type not supported!")       