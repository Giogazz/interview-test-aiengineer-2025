### Report Giovanni Gazzola

------

#### How to run

In the project foder run:

```
./run.sh --n_estimators 5 --n_jobs 2 --data_size 10000 --use_author
```

where the parameters are:

- `n_estimators`: number of tree of the random forest;
- `n_jobs`: cores used to train the model;
- `data_size`: how many instances create before train the model;
- `use_author`: to be passed if train with author information;



#### Data Exploration and Data Cleaning

Before editing the main.py script I create a jupiter notebook file to explore the data. 
Three different dataset are available. I decided to use just one of them, in particular the one with medium size: `arcan-2-develop-none.csv`.

In this initial phase I load the dataset and compute some basic statistics over it:

- Total number of rows;
- Total number of columns;
- If there were any NaN values;
- Number of uniques values for each column;
- If any file has been renamed with a commit (e.g. different filenames between 'old_file' and 'new_file' columns);

I decide to drop some columns to start with an easy scenario to get a baseline model. 
Columns 'old_file' and 'new_file' were identical so, I use just one of the two. 
The 'new_author' column has been selected instead of  the 'old_author' column because I find more informative who is making the commit instead of who was the last person that changed the file.

The remaining columns are: 'parent_sha', 'child_sha', 'old_file', 'new_author'.


I use 'parent_sha' and 'child_sha' values to group the files that have been modified in the same commit. After that, I drop the the 'parent_sha' and 'child_sha' columns.

What I have at the end is a list of files for each commit and the author of the commit.

To conclude the data exploration phase I plot two heatmap.
This plots are used to check (in a very basic way) if some files are more correlated than others and if some authors tend to work more on the same files.
The resulting plots are not very intuitive. Anyway some brighter area can be seen in them, meaning that some sort of correlation between files-files and files-author could exists.

**All these filtering steps are then repeated in the main.py script.**



#### Data representation

At this point I have groups of files which have been committed together + the commit author.

One-hot encoding vectors are used to represent each file and each author. 
This representation is one of the easiest. With it,  represent multiple files in the same compact vector is pretty straightforward due to the vector orthogonality (simple sum). 
For a single file I have a vector of dimension 3835 (number of unique files) where all the elements are zero except the one at the *i-th* index, which is the index that has been assigned to the considered file.
Same is done for the authors, where we have vectors of 4 dimensions (for simplicity the authors have been group by name passing from 8 to 4, e.g. 'Darius Daniel Sas' and 'Darius Sas' --> 'Darius').



#### Dataset creation

The goal is to have a model that, given *N* input files, predicts *T* output files that probably required some modification.
Before create my training/test instances I compute the probability that each single file is modified in a commit. A file that appears in more commit is a file that is edited more than the others. I want to preserve the original data distribution.

After that I do the following steps:

- given *n* files in a commit I select a random number, lets call it *s*, between 1 and *n*-1;
- i randomly  sampled *s* files from the commit  based on their probability to be modified, sum their associated sparse vectors and obtain *x* (input sample);
- create the *y* (target vector) by summing up the remaining *n-s* vectors associated to the files in the commit that have not been included in the *x* vector;
- repeat the process for each commit and until a maximum number of instances is reached;

The same process is used to include the author representation. The only difference is that the author vector is appended to *x* and not in *y* (we just want to predict the files).
At the end we have:

- *x*: sparse vector of dimension 3835 without author information or 3839 with author information;
- *y*: sparse vector of dimension 3835; 

The data is then splitted in train and test. Test is 10% of the total number of instances.



#### Model Training and Test

Given scikit-learn as python ML library, the model trained is a **Random Forest**.
The main reasons are the multi-label classification problem and the non-linearity nature of the problem (assumption).

Two experiments are run:

- without authors information in the training instances;

- with authors information in the training instances;



The evaluation metrics used  is the *log_loss* of scikit_learn which corresponds to the *CrossEntropyLoss*.
This metrics tells us how confident the model is to predict if a file needs to be modified or not given the input files.  A low loss value is better.

For each prediction of the model, over the test set, the loss is computed and then the average between all the losses is performed. 

The results obtained, with *n_estimators=5* and *data_size=10000*, for the two experiments are:

- Average log loss without authors:
- Average log loss with authors:



The added authors information seems to help the model in the predictions.



#### Limitation and possible implementation

1) The use of just a portion of the available data columns. More features can help the model increase its performance, as has been shown for the authors information.
2) One-hot encoding for files representation. This representation do not consider the location of the file inside the repo structure. We are losing some useful information.
3) One-hot encoding for files representations is not suitable to handle new unseen files. The "vocabulary" of the trained model is fixed.
4) A more complex network such as a Feed Forward Neural Network or a Transformer can better handle the complexity of the problem.

