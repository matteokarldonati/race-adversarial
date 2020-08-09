import numpy as np
import pandas as pd

import torch


def accuracy(df):
    return (df['pred'] == df['label']).mean()


def adv_accuracy(df):
    num_perturbated = df[df['run'] == 0]['perturbated'].sum()

    n = int(df['run'].max()) + 1

    correct_pred = (df[df['run'] == 0]['pred'] == df[df['run'] == 0]['label']).values

    for i in range(1, n):
        condition = (df[df['run'] == (i - 1)]['label'].values == df[df['run'] == i]['label'].values).sum()
        assert ((condition == 4934) or (condition == num_perturbated))

        correct_pred = correct_pred & (df[df['run'] == i]['pred'] == df[df['run'] == i]['label']).values

    return correct_pred.mean()


def get_df_results(output_path):
    df = pd.DataFrame()

    pred = torch.load(output_path + 'test_predictions')
    labels = torch.load(output_path + 'test_labels_id')
    examples_ids = torch.load(output_path + 'examples_ids')
    perturbated = torch.load(output_path + 'perturbated')
    run = torch.load(output_path + 'run')

    group = []
    for i in examples_ids:
        group.append(i.split('/')[-2])

    df['group'] = group
    df['pred'] = np.argmax(pred, axis=1)
    df['label'] = labels
    df['perturbated'] = perturbated
    df['run'] = run

    return df


def get_results(output_path):
    df = get_df_results(output_path)

    print('#### Results Original Test Set ####')
    print('Accuracy: ', accuracy(df[df['run'].isna()]))
    print('Accuracy Perturbated: ', accuracy(df[(df['run'].isna()) & df['perturbated'] == True]))
    print('===================================')

    runs = []

    n = int(df['run'].max()) + 1

    for i in range(n):
        runs.append(accuracy(df[df['run'] == i]))

    runs = np.array(runs)

    print('#### Multiple Runs Results ####')
    print('Mean Accuracy: ', runs.mean())
    print('Standard Deviation: ', runs.std())
    print('===================================')

    runs_perturbated = []

    for i in range(n):
        runs_perturbated.append(accuracy(df[(df['run'] == i) & df['perturbated'] == True]))

    runs_perturbated = np.array(runs_perturbated)

    print('#### Multiple Runs Perturbated Results ####')
    print('Mean Accuracy: ', runs_perturbated.mean())
    print('Standard Deviation: ', runs_perturbated.std())
    print('===================================')

    print('#### Multiple Runs Perturbated Subset Results ####')
    print('Adversarial Accuracy :', adv_accuracy(df))
    print('Adversarial Accuracy Perturbated: ', adv_accuracy(df[df['perturbated'] == True]))
    print('===================================')
