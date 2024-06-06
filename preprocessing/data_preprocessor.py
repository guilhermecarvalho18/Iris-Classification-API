import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import KNNImputer 
from imblearn.over_sampling import SMOTE
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    
    def __init__(self, filepath, header=None, names=None, missing_threshold=0.7, n_neighbors=5, variance_threshold=0.0):
        self.filepath = filepath
        self.header = header
        self.names = names
        self.data = None
        self.scaler = StandardScaler()
        self.missing_threshold = missing_threshold
        self.imputer = KNNImputer(n_neighbors=n_neighbors)
        self.variance_threshold = variance_threshold


    def load_data(self, **kwargs):
        self.data = pd.read_csv(self.filepath, 
                                header=self.header, 
                                names=self.names)
        return self
    

    def handle_missing_values(self):
        missing_percentages = self.data.isnull().mean()
        columns_to_drop = missing_percentages[missing_percentages > self.missing_threshold].index
        self.data.drop(columns = columns_to_drop, 
                       inplace = True)

        if self.data.isnull().values.any():
            self.data = pd.DataFrame(self.imputer.fit_transform(self.data), 
                                     columns=self.data.columns)
        
        return self
    

    def normalize_features(self, method='standard'):
        target_column = 'species'
        if target_column in self.data.columns:
            features = self.data.drop(columns=target_column)
        else:
            features = self.data

        if method == 'standard':
            self.data[features.columns] = self.scaler.fit_transform(features)
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
            self.data[features.columns] = self.scaler.fit_transform(features)
        return self    


    def handle_imbalance(self):
      if 'class' in self.data.columns:
        X = self.data.drop(columns='class')
        y = self.data['class']
        if y.value_counts().min() / y.value_counts().max() < 0.5:  # If imbalance ratio is significant
            smote = SMOTE()
            X_res, y_res = smote.fit_resample(X, y)
            self.data = pd.concat([X_res, y_res], axis=1)

      return self
    

    def filter_noise(self, z_threshold=3):
        z_scores = stats.zscore(self.data.select_dtypes(include=[float, int]))
        abs_z_scores = abs(z_scores)
        filtered_entries = (abs_z_scores < z_threshold).all(axis=1)
        self.data = self.data[filtered_entries]
        return self


    def apply_pca(self, n_components=2, variance_threshold=0.95):
        if 'class' in self.data.columns:
            features = self.data.drop(columns='class')
        else:
            features = self.data

        pca = PCA(n_components = n_components)
        pca.fit(features)
        explained_variance = sum(pca.explained_variance_ratio_)
        
        if explained_variance < variance_threshold:
            principal_components = pca.transform(features)
            pc_df = pd.DataFrame(data=principal_components, 
                                 columns=[f'PC{i+1}' for i in range(n_components)])
            self.data = pd.concat([pc_df, self.data['class']], 
                                  axis=1)

        return self
    

    def select_features(self, k='all'):
        if 'species' in self.data.columns:
            X = self.data.drop(columns='class')
            y = self.data['species']
        else:
            X = self.data
            y = None

        if k == 'all':
            return self
            
        if k > X.shape[1]:
            print(f"Warning: k ({k}) is greater than the number of features ({X.shape[1]}). Using all features.")
            return self
            
        selector = SelectKBest(score_func=chi2, k=k)
        selected_features = selector.fit_transform(X, y)
        selected_columns = X.columns[selector.get_support()]
        self.data = pd.concat([pd.DataFrame(selected_features, 
                                            columns=selected_columns), 
                                            y], 
                                            axis=1)
        return self


    def encode_labels(self, column='class', method='label', drop_first=True):
        if column in self.data.columns:
            if method == 'label':
                if self.data[column].dtype == 'object':
                    encoder = LabelEncoder()
                    self.data[column] = encoder.fit_transform(self.data[column])

            elif method == 'onehot':
                if self.data[column].dtype == 'object':
                    encoder = OneHotEncoder(drop=drop_first, 
                                            sparse=False)
                    encoded_features = encoder.fit_transform(self.data[[column]])
                    encoded_df = pd.DataFrame(encoded_features, 
                                            columns=encoder.get_feature_names_out([column]))
                    self.data = self.data.drop(columns=[column]).join(encoded_df)
                
        return self
    
    def save_data(self, filepath, index=False, encoding='utf-8'):
        self.data.to_csv(filepath, 
                         index=index, 
                         encoding=encoding)
        
