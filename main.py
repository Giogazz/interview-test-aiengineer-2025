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

def comput_prob(instances, file_to_id):
    #count occurencies
    prob_mod_file = {}
    for instance in instances:
        for file in instance:
            index = file_to_id[file]
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

def author_data_prep(df):
    #collect unique author names
    unique_authorname = sorted(list(set(df['new_author'])))
    #get author per instance
    df_author_name = df['new_author'].tolist()

    #map author names
    unique_authorname_map = {
        'Alessandro Tundo': 'Alessandro',
        'Darius': 'Darius',
        'Darius Daniel Sas': 'Darius',
        'Darius Sas': 'Darius',
        'Ilaria Pigazzini': 'Ilaria',
        'Luca': 'Luca',
        'Luca Belluzzi': 'Luca',
        'LucaArcan': 'Luca'
    }
    #re-write author per instances with the map applied
    author_mapped_instances = [unique_authorname_map[name] for name in df_author_name]
    #create author_to_id
    author_to_id = {name: index for index, name in enumerate(sorted(list(set(author_mapped_instances))))}

    return author_mapped_instances, author_to_id

def create_training_instances_with_author(instances, one_hot_file, unique_filepath, prob_mod_file, file_to_id, data_size, author_mapped_instances, author_to_id, sparse_vectors_author):
    #randomly create input instance and target instance
    x = np.empty([data_size, len(unique_filepath) + len(sparse_vectors_author)], dtype=int)
    y = np.empty([data_size, len(unique_filepath)], dtype=int)

    c = 0
    while c < data_size-1:
        for author, instance in zip(author_mapped_instances, instances):
            if len(instance) == 1: continue
            elif len(instance) == 2:
                num_file_mod = 1
            else:
                num_file_mod = np.random.randint(1, len(instance)-1)

            ids = [file_to_id[file] for file in instance]
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
            x[c] = np.hstack([tmp_x, sparse_vectors_author[author_to_id[author]].reshape(1, -1)])
            y[c] = tmp_y
            c+=1
            if c > data_size-1: break

    print(f"All training instances have at least one mod file: {np.any(np.sum(x, axis=1)>=1)}")
    print(f"All target instances have at least one mod file: {np.any(np.sum(y, axis=1)>=1)}")
    return x, y

def create_training_instances(instances, one_hot_file, unique_filepath, prob_mod_file, file_to_id, data_size):
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

            ids = [file_to_id[file] for file in instance]
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

def split_data(x, y, data_size, n_test):
    #split data
    X_train = x[0:data_size-n_test]
    Y_train = y[0:data_size-n_test]
    X_test = x[data_size-n_test::]
    Y_test = y[data_size-n_test::]

    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_test shape: {Y_test.shape}")

    #ensure eache feature in the Y_train have at least one non-zero
    if np.all(np.sum(Y_train, axis=0) >= 1) == False:
        to_change = np.where(np.sum(Y_train, axis=0) == 0)
        for col in to_change:
            Y_train[np.random.randint(0, Y_train.shape[0]-1), col] = 1

    print(f"All training features have been mod at leas on time: {np.all(np.sum(Y_train, axis=0) >= 1)}")
    return X_train, Y_train, X_test, Y_test

def model_train(X_train, Y_train, n_estimators, n_jobs):
    #initialize the model
    rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)

    #train the model
    rf.fit(X_train, Y_train)
    return rf

def test_model(rf, X_test, Y_test):
    #compute model predictions
    y_out = rf.predict_proba(X_test)
    #take just the probabilty that the files need to be edited
    y_out = np.array([probs[:, 1] for probs in y_out]).T

    #compute the CrossEntropyLoss for all the model prediction
    losses = [log_loss(Y_test[i, :], y_out[i, :], labels=[0, 1]) for i in range(Y_test.shape[0])]

    #compute the mean
    avg_loss = np.mean(losses)
    print(f"Average Log Loss: {avg_loss}")

    return y_out, losses

def show_example(y_out, Y_test, losses, file_to_id, threshold):
    #collect model prediction in a more readable way
    file_to_mod = []
    for pred in y_out:
        sugg_mod_file = {list(file_to_id.keys())[index]: pred[index].item() for index in np.where(pred >= threshold)[0]}
        file_to_mod.append(sugg_mod_file)

    worst = losses.index(max(losses))
    best = losses.index(min(losses))

    print(f"\nLower loss (best) model prediction")
    ref_list = [list(file_to_id.keys())[index] for index in np.where(Y_test[best] != 0)[0]]
    print(f"\nReference files: {ref_list}")
    print(f"\nPredicted files: {file_to_mod[best]}")

    print(f"\n\nHigher loss (worst) model prediction")
    ref_list = [list(file_to_id.keys())[index] for index in np.where(Y_test[worst] != 0)[0]]
    if len(ref_list) <= 10:
        print(f"\nReference files: {ref_list}")
    else:
        print(f"\nReference files truncated: {ref_list[:5]}...{ref_list[-5:]}")
    print(f"\nPredicted files: {file_to_mod[worst]}")


def main(data_path):
    print("\n\nSTARTING\n\n")
    #read arguments
    n_estimators = args.n_estimators
    n_jobs = args.n_jobs
    if n_jobs <= 0: n_jobs = -1
    data_size = args.data_size
    if data_size <=0 : print(f"Cannot create a dataset with {data_size} rows."); sys.exit()
    use_author = args.use_author
    dry_run = args.dry_run
    csv_path = '/data/arcan-2-develop-none.csv'

    if dry_run: check_works()

    #import csv file
    print(f"Reading dataset: {csv_path}")
    df = import_dataset(csv_path)
    print(f"Total dataset rows: {len(df)}")
    print(f"Total dataset columns: {len(df.columns)}")

    #filtering the dataset
    print(f"\nFilter the dataset")
    df_selected = filter_dataset(df)

    #get unique filepath
    unique_filepath = sorted(list(set(df['old_file'])), key=lambda x: len(x.split('/')))
    num_filepath = len(unique_filepath)
    print(f"Numbers of unique files: {num_filepath}")

    #map files to ids
    file_to_id = {file: index for index,file in enumerate(unique_filepath)}
    #create one-hot encoding for the filepath
    one_hot_file = one_hot(len(unique_filepath))
    #get files insatnces
    instances = df_selected['old_file'].tolist()
    #compute the probability of a file to be changed when a commit is done
    prob_mod_file = comput_prob(instances, file_to_id)

    if use_author:
        #pre-proc author column
        author_mapped_instances, author_to_id = author_data_prep(df)
        #get one-hot representation for authors
        sparse_vectors_author = one_hot(len(author_to_id))

        #create input instances and thier target vector with author information
        print(f"\n\nCreating {data_size} instances to train the model with author information")
        x, y = create_training_instances_with_author(instances, one_hot_file, unique_filepath, prob_mod_file, file_to_id, data_size, author_mapped_instances, author_to_id, sparse_vectors_author)
    else:
        #create input instances and their respective target
        print(f"\n\nCreating {data_size} instances to train the model")
        x, y = create_training_instances(instances, one_hot_file, unique_filepath, prob_mod_file, file_to_id, data_size)

    print(f"Split in train and test sets")
    X_train, Y_train, X_test, Y_test = split_data(x, y, data_size, n_test=int(0.1*data_size))

    print(f"\n\nTraining RadomForest with n_estimators={n_estimators} and n_jobs={n_jobs}")
    rf = model_train(X_train, Y_train, n_estimators, n_jobs)
    print("Done\n")

    print(f"Test model performance over test_set")
    y_out, losses = test_model(rf, X_test, Y_test)

    print("\nShowing some predictions examples, where decision threshold is 50%")
    show_example(y_out, Y_test, losses, file_to_id, 0.5)

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
