import glob
import numpy
import pandas
import sklearn

csv_files = glob.glob("/data/*.csv")

if len(csv_files) > 0:
    print("Yay! I can read the data:")
    print(csv_files)
else:
    print("Darn! I cannot read the data at /data.")
    print("Probably because the ./data dir on the host is not mounted correctly to /data in the container.")

