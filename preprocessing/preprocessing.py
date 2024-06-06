from preprocessing.data_preprocessor import DataPreprocessor


data = DataPreprocessor('data/raw/iris.csv',
                  header = None,
                  names = ['sepalLength', 
                           'sepalWidth', 
                           'petalLength', 
                           'petalWidth',
                           'class'])

df = data.load_data()

df = df.handle_missing_values()
df = df.normalize_features()
df = df.handle_imbalance()
df = df.filter_noise()
df = df.apply_pca()

df.save_data('data/iris.csv')







