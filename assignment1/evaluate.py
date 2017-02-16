from __future__ import print_function
from argparse import ArgumentParser
import sys

try:

    import numpy as np
    import pandas as pd

except:

    print("numpy and pandas are required!")

    print("pip install numpy / conda install numpy")
    print("pip install pandas / conda install pandas")

    sys.exit(1)


# Argument parsing:

parser = ArgumentParser()

parser.add_argument("-p", "--predict", dest = "predict_path",
    required = True, help = "path to your model's predicted labels file")

parser.add_argument("-d", "--development", dest = "dev_path",
    required = True, help = "path to the development labels file")

args = parser.parse_args()


# Load predicted and dev CSV files:

predict = pd.read_csv(args.predict_path).set_index("id")
dev = pd.read_csv(args.dev_path).set_index("id")

predict.columns = ["predicted"]
dev.columns = ["actual"]


# Join observations by ID:

data = dev.join(predict)


# Calculate accuracy and count missing obervations:

accuracy = (data["actual"] == data["predicted"]).mean()
missing = pd.isnull(data["predicted"]).sum()

print("Accuracy:", accuracy)

if missing > 0:

    print("There are", missing, "missing observations!")
