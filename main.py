import glob
import sys
import numpy as np
import random
import pandas as pd
import sklearn
import argparse
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


def check_works():
    csv_files = glob.glob("/data/*.csv")

    if len(csv_files) > 0:
        print("Yay! I can read the data:")
        print(csv_files)
    else:
        print("Darn! I cannot read the data at /data.")
        print("Probably because the ./data dir on the host is not mounted correctly to /data in the container.")

    sys.exit()

def import_dataset(csv_path):
    return  pd.read_csv(csv_path)

def filter_dataset(df):
    #drop usless columns
    drop_columns = ['new_file', 'old_lines', 'new_lines', 'old_lines', 'old_author','when']
    print(f"Dropping some columns: {drop_columns}")
    df = df.drop(drop_columns, axis=1)

    #group files that changed with the same commit and the commit author
    df = df.groupby(['parent_sha','child_sha'], group_keys=False, as_index=False).agg({'old_file': list, 'new_author': 'first'})
    #now drop parent_sha and parent_sha columns
    return df.drop(['parent_sha', 'child_sha'], axis=1)

def comput_prob(instances, id_to_file):
    #count occurencies
    prob_mod_file = {}
    for instance in instances:
        for file in instance:
            index = id_to_file[file]
            if index not in prob_mod_file.keys():
                prob_mod_file[index] = 1
            else:
                prob_mod_file[index] = prob_mod_file[index] + 1

    #divide by the total number of commits
    total_commit = len(instances)
    for key in prob_mod_file:
        prob_mod_file[key] = prob_mod_file[key]/total_commit

    return prob_mod_file

def one_hot(num_features):
    #convert each file into a sparse vector
    sparse_vectors = np.diag(np.full(num_features, 1))
    return sparse_vectors

def create_training_instances(instances, one_hot_file, unique_filepath, prob_mod_file, id_to_file, data_size):
    #randomly create input instance and target instance
    x = np.empty([data_size, len(unique_filepath)], dtype=int)
    y = np.empty([data_size, len(unique_filepath)], dtype=int)

    c = 0
    while c < data_size-1:
        for instance in instances:
            if len(instance) == 1: continue
            elif len(instance) == 2:
                num_file_mod = 1
            else:
                num_file_mod = np.random.randint(1, len(instance)-1)

            ids = [id_to_file[file] for file in instance]
            file_mod = 0
            tmp_x = np.zeros([1, len(unique_filepath)])
            tmp_y = np.zeros([1, len(unique_filepath)])
            while file_mod < num_file_mod:
                id = np.random.choice(ids)
                if random.random() > prob_mod_file[id]:
                    tmp_x = tmp_x + one_hot_file[id]
                    file_mod += 1
                    ids.remove(id)

            assert len(ids) > 0
            tmp_y = np.sum(one_hot_file[ids],axis=0)
            x[c] = tmp_x
            y[c] = tmp_y
            c+=1
            if c > data_size-1: break

    print(f"All training instances have at least one mod file: {np.any(np.sum(x, axis=1)>=1)}")
    print(f"All target instances have at least one mod file: {np.any(np.sum(y, axis=1)>=1)}")
    return x, y

def main(data_path):
    #read arguments
    n_estimators = args.n_estimators
    n_jobs = args.n_jobs
    data_size = args.data_size
    use_author = args.use_author
    dry_run = args.dry_run
    csv_path = '/data/arcan-2-develop-none.csv'

    if dry_run: check_works()

    #import csv file
    df = import_dataset(csv_path)
    print(f"Total dataset rows: {len(df)}")
    print(f"Total dataset columns: {len(df.columns)}")

    df_selected = filter_dataset(df)

    #get unique filepath
    unique_filepath = sorted(list(set(df['old_file'])), key=lambda x: len(x.split('/')))
    num_filepath = len(unique_filepath)
    print(f"Numbers of unique files: {num_filepath}")

    #map files to ids
    id_to_file = {file: index for index,file in enumerate(unique_filepath)}
    #create one-hot encoding for the filepath
    one_hot_file = one_hot(len(unique_filepath))

    instances = df_selected['old_file'].tolist()
    #compute the probability of a file to be changed when a commit is done
    prob_mod_file = comput_prob(instances, id_to_file)

    #create input instances and their respective target
    print(f"Creating {data_size} instances to train the model")
    x, y = create_training_instances(instances, one_hot_file, unique_filepath, prob_mod_file, id_to_file, data_size)




if __name__ == "__main__":
    #initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', '-ne', type=int, default=5, help='Number of estimator for the RandomForestClassifier')
    parser.add_argument('--n_jobs', '-j', type=int, default=2, help='Number of jobs to train the RandomForestClassifier; -1 to use all cores.')
    parser.add_argument('--data_size', '-d', type=int, default=10000, help='Number of instances to create before train the model.')
    parser.add_argument('--use_author', '-a', action="store_true", help='If passed it encodes also the author name to make predictions')
    parser.add_argument('--dry_run', '-dr', action="store_true", help='Check everything works properly.')

    args = parser.parse_args()
    main(args)
