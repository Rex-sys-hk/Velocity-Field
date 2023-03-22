import argparse
import pandas as pd
import numpy as np
import glob
from common_utils import cellect_results
import os

def static_result(df,name, close_loop = False):
    result = {}
    for k in df.keys():
        if k != 'Unnamed: 0':
            result[k] = np.mean(df[k])
    print(result)
    df = pd.DataFrame(data=result, index=[0])
    csv_name = f'./testing_log/{name}/testing_log_stat.csv' if not close_loop else f'./testing_log/{name}/testing_log_stat_cl.csv'
    df.to_csv(csv_name)
    print(f'Results saved in:\n {csv_name}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name (default: "Exp1")', default="Exp1")
    parser.add_argument('--cl', action='store_true', help='log name (default: "Exp1")', default=False)
    args = parser.parse_args()
    files = glob.glob(f'./testing_log/{args.name}/*')
    # print(len(files))
    # csv_name = f'./testing_log/{args.name}/testing_log.csv' if not args.cl else f'./testing_log/{args.name}/testing_log_cl.csv'
    results = []
    for csv_name in files:
        name, types=os.path.splitext(csv_name)
        # print(name, types)
        if types == '.csv':
            df = pd.read_csv(csv_name)
            results.append(df.to_dict('List'))

    df = cellect_results(results)
    # data = dict(zip(df['id'], df['values']))
    static_result(df, args.name, args.cl)