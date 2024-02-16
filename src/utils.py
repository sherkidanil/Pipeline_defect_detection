import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

def visualize_pipe(id: str, df: pd.DataFrame,
                   df_with_targets: pd.DataFrame = None):
    df = df[df['id']==id]
    
    plt.plot(df['time'], df['ch0'], label = 'channel 0')
    plt.plot(df['time'], df['ch1'], label = 'channel 1')
    plt.plot(df['time'], df['ch2'], label = 'channel 2')
    if df_with_targets is not None:
        target = df_with_targets.loc[df_with_targets['id'] == id]['target'].iloc[0]
        title = f'Pipe {id} with {target} defect'
    else:
        title = f'Pipe {id}'
    plt.title(title)
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('channel value')

def calculate_metrics(X_test, y_test, pipeline):
    
    accuracy = pipeline.score(X_test, y_test)
    y_pred = pipeline.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'Accuracy: {accuracy}, F1 score: {f1}')

def train_pipeline(X_train, y_train, X_test, y_test, classifier):

    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', classifier)
    ])
    pipeline.fit(X_train, y_train)

    print('For train_set:')
    calculate_metrics(X_train, y_train, pipeline)
    print('For test set:')
    calculate_metrics(X_test, y_test, pipeline)
    return pipeline

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    stats_per_id = df[['id','ch0', 'ch1', 'ch2']].groupby('id').describe()
    result = pd.DataFrame({'id':df['id'].unique()})
    for col in ['ch0', 'ch1', 'ch2']:
        for prop in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
            buf = pd.merge(result, stats_per_id[col][prop], on='id')
            result[f'{col}_{prop}'] = buf[prop]
    return result


def visualize_log(logs: np.array, log_scale: bool = False):
    plt.figure(figsize = (12,6))
    plt.plot(logs[:, 0], logs[:, 1], label = 'train')
    plt.plot(logs[:, 0], logs[:, 2], label = 'validation')
    plt.title("Loss curves")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    if log_scale:
        plt.yscale('log')
    plt.show()