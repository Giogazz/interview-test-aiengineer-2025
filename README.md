# Technical interview
This repository contains the starting files for the technical interview for the AI Engineer applications at TXT Arcan.

Applicants should fork this repository, complete the test, and submit the link to their solution to the hiring manager/HR contact.

For any quetions related to the test, please contact `darius.sas _at_ arcan.tech`.

## The task
### Goal
The goal is to *study*, design and implement a machine learning model that predicts change propagation in a software repository.

Change propagation is loosely defined as a file that is forced to change due to another file changing.

It's typically caused by [coupling](https://en.wikipedia.org/wiki/Coupling_%28computer_programming%29) (either logical or static). For instance, a function changes its signature, and, as a consequence, all the files in the system that use that function need to be updated. 

Therefore, given a list of files `Xs`, the model must output a list of files `Ys` that are likely to change.
How exactly this is defined (and thus how the model is evaluated) is within the scope of this test and you are welcome to interpret it however you see fit.

### Expected output

- A brief report (max 1000 words) explaining the key technical decisions taken to clean the data, design the ML model, and train and evaluate the model. You can save the report and all non-source code files must be in the `report` folder. You can stick to an informal structure, like bullet points, if you wish.
- A *dockerized* Python script that trains the ML model and evaluates it using the data provided.

Please do not spend more than **4 hours** of work on this task. The report and the code have an equal weight during the evaluation of your submission.
However, do prioritize the report in case you do not manage to finish the code, detailing the actions you planned to implement in the script.

### The data
The `./data` folder contains data mined from a few repositories.

Each CSV file contains the `git diffs` mined from a single repository. Every file has the following header:

- `parent_sha` the sha of the parent commit
- `child_sha` the sha of the child commit
- `old_file` the name of the file that changed in the parent commit
- `new_file` the name of the file that changed in the child commit (if it differs from `old_file` it means that the file was renamed
- `old_lines` number of lines deleted
- `new_lines` number of lines added/modified
- `old_author` author of the parent commit
- `new_author` author of the child commit
- `when` the date of the child commit

Therefore, every "diff" represents the files that changed from one commit to the next. 
For simplicity's sake, you can ignore renames and deletions.

The data belongs to three repositories:

- `arcan-2-develop-none.csv` diffs of Arcan2's develop branch. Contains 3836 files.
- `ffmpeg-master-none.csv` diffs of open source project FFmpeg's master branch of 2024. Contains 4417 files.
- `antlr4-dev-none.csv` diffs of open source project Antlr4's dev branch. Contains 9528 files.

Feel free to filter out files with few changes or to **focus only on one** repository.


## Instructions
The repository contains a basic configuration that you can use as a starting point.

Unzip the data with the following commands
```bash
cd data
tar xzf diffs.tar.gz
```

then run the container using the handy `./run.sh`.
You need Docker installed.

The script will build the container and run `python main.py`. Any arguments passed to `run.sh` will be forwarded to your `main.py` script.
