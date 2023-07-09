import sys
from sklearn.model_selection import KFold
import os
import pandas as pd

def kfold(dataset, folds=5, iszip=False):

    def old2new(old, type):
        new = old.replace('.csv', '').replace('.gz', '').replace('.xz', '')
        new = new.replace('/quantise/', '/cv/{0}/quantise/'.format(type))
        #new = new.replace('/origin/', '/cv/{0}/origin/'.format(type))
        new = new.replace('/complete/', '/')
        return new

    if dataset.endswith('xz'):
        comprs = 'xz'
    elif dataset.endswith('gz'):
        comprs = 'gzip'
    else:
        comprs = None

    df = pd.read_csv(dataset, compression=comprs)

    kf = KFold(n_splits=folds, shuffle=True, random_state=2)

    i = 0

    for train_index, test_index in kf.split(df):
        i += 1
        t2index = {'train': train_index, 'test': test_index}

        for t in ['train', 'test']:
            if '_data.csv' in dataset:
                new_file = dataset.replace('_data.csv', f'_{t}{i}_data.csv')
            else:
                new_file = dataset.replace('.csv', f'_{t}{i}.csv')
            new_file = new_file.replace('/complete/', f'/{t}/')
            saved_dir = new_file.rsplit('/', maxsplit=1)[0]
            if not os.path.isdir(saved_dir):
                os.makedirs(saved_dir)
            new_df = df.iloc[t2index[t], :]
            new_df.to_csv(new_file, index=False)

            if '_data.csv' in dataset:
                os.system('cp {0}.pkl {1}.pkl'.format(dataset, new_file))

if __name__ == '__main__':
    datasets = []
    for root, dirs, files in os.walk('../datasets/complete'):
        for file in files:
            if file.endswith('.csv') and '_discrete' not in file:
                datasets.append(os.path.join(root, file))

    for dataset in datasets:
        kfold(dataset)
