import json
import os
import random

DATA_PATH = ".\\data"
DATASETS = ["1", "2", "3", "4", "A", "B", "C", "D"]
MAPPINGS = {"1": [1,0,0,0,0,0,0,0],
            "2": [0,1,0,0,0,0,0,0],
            "3": [0,0,1,0,0,0,0,0],
            "4": [0,0,0,1,0,0,0,0],
            "A": [0,0,0,0,1,0,0,0],
            "B": [0,0,0,0,0,1,0,0],
            "C": [0,0,0,0,0,0,1,0],
            "D": [0,0,0,0,0,0,0,1],}
TRAIN_PERCENTAGE = 0.8
TEST_PERCENTAGE = 1 - TRAIN_PERCENTAGE

TRAINING_SET_NAME = "#train_{}%.json".format(round(TRAIN_PERCENTAGE * 100))
TEST_SET_NAME = "#test_{}%.json".format(100 - round(TRAIN_PERCENTAGE * 100))


# representing data in pairs (x, y)
# x is a 1200 (flattened) sequence and y is a 1 * 8 one-hot vector
all_data = []

for ds_name in DATASETS:
    ds_path = os.path.join(DATA_PATH, ds_name+"_uniformed.json")
    with open(ds_path, "r") as dsfp:
        ds = json.load(dsfp)
        for seq in ds:
            # flat = []
            # for attrib in seq:
            #     flat += attrib
            # all_data.append((flat, MAPPINGS[ds_name]))
            all_data.append((seq, MAPPINGS[ds_name]))

# shuffle data
random.shuffle(all_data)

# split data
train_size = round(TRAIN_PERCENTAGE * len(all_data))
training_set = all_data[:train_size]
test_set = all_data[train_size:]

# save
with open(os.path.join(DATA_PATH, TRAINING_SET_NAME), "w") as fp:
    json.dump(training_set, fp)
with open(os.path.join(DATA_PATH, TEST_SET_NAME), "w") as fp:
    json.dump(test_set, fp)
    
