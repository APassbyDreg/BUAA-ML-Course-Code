import os
import json


SOURCE_PATH = ".\\original\\raw_data\\"
DATA_PATH = ".\\data"
DATASETS = ["1", "2", "3", "4", "A", "B", "C", "D"]

for ds_name in DATASETS:
    ds_path = os.path.join(SOURCE_PATH, ds_name)
    for root, dirs, files in os.walk(ds_path):
        ds = {}
        for fn in files:
            if fn.endswith(".txt"):
                f = open(os.path.join(ds_path, fn), 'rb')
                lines = [l.decode("utf-8").strip() for l in f.readlines()]
                # decode and store in list (size: 7 * length)
                data_seq = {"timestamp": [], 
                            "speed_accel": [],
                            "angle_accel": []}
                for i in range(len(lines) // 4):
                    timestamp_str = lines.pop(0)
                    lines.pop(0) # skip useless lines
                    speed_accel_str = lines.pop(0)
                    angle_accel_str = lines.pop(0)
                    timestamp = float(timestamp_str.split("  ")[1])
                    speed_accel = [float(i) for i in speed_accel_str[4:].split("  ")]
                    angle_accel = [float(i) for i in angle_accel_str[4:].split("  ")]
                    data_seq["timestamp"].append(timestamp)
                    data_seq["speed_accel"].append(speed_accel)
                    data_seq["angle_accel"].append(angle_accel)
                ds[fn] = data_seq
        with open(os.path.join(DATA_PATH, ds_name+".json"), "w") as fp:
            json.dump(ds, fp)
        print("loaded dataset \'{}\' to json".format(ds_name))