import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.model_selection import train_test_split
from preprocessing.data_preprocessor import DataPreprocessor
from model.model_trainer import ModelTrainer


data = DataPreprocessor('data/raw/iris.csv',
                  header = None,
                  names = ['sepalLength', 
                           'sepalWidth', 
                           'petalLength', 
                           'petalWidth',
                           'class'])
data.load_data()
data.encode_labels()

X = data.data.drop(columns='class')
y = data.data['class']

# Split the data into train and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size =0.3, 
                                                    random_state=42, 
                                                    stratify = y)

"""
# Split the train set further into train and validation sets (20% of train set as validation set)
X_train, X_val, y_train, y_val = train_test_split(X_train, 
                                                  y_train, 
                                                  test_size=0.2, 
                                                  random_state=42, 
                                                  stratify=y_train)
"""

trainer = ModelTrainer(X_train, y_train, X_test, y_test)
trainer.train_and_evaluate_all()

