import torch

import os
import json

import Loader as myds
from DNN import *
from LSTM import *
from EnsembleLearning import *


MODEL_PATH = ".\\models\\train_from_extended"
TAGS = ["1", "2", "3", "4", "A", "B", "C", "D"]


def __evaluate(model_name, x_all, y_all):
    # load given model
    model = torch.load(os.path.join(MODEL_PATH, model_name))

    # evaluate
    ## fails[i][0]: count items in other class, misclassified to this class
    ## fails[i][1]: count items in this class, misclassified to other class
    fails = {i: [0, 0] for i in TAGS}     
    success = {i: 0 for i in TAGS}
    total = {i: 0 for i in TAGS}
    fail_details = []
    with torch.no_grad():
        predict = model(x_all)
        for idx in range(y_all.shape[0]):
            tag_id = torch.argmax(y_all[idx], dim=-1)
            predict_id = torch.argmax(predict[idx], dim=-1)
            total[TAGS[tag_id]] += 1
            if tag_id != predict_id:
                fails[TAGS[predict_id]][0] += 1
                fails[TAGS[tag_id]][1] += 1
                # record related info
                fail_info = {"input_hash": hash("".join(map(str ,x_all[idx].view(-1).tolist()))),
                             "model_output": predict[idx].tolist(),
                             "target_output": tag_id.tolist()}
                fail_details.append(fail_info)
            else:
                success[TAGS[tag_id]] += 1

    # calculate accuracies
    output_accuracy = {tag: (success[tag] / (success[tag] + fails[tag][0])) for tag in TAGS}
    class_accuracy = {tag: (1 - fails[tag][1] / total[tag]) for tag in TAGS}
    overall_accuracy = sum(success.values()) / y_all.shape[0]

    result = {"model-name": model_name,
              "accuracy-overall": overall_accuracy,
              "accuracy-per-class": class_accuracy,
              "accuracy-on-output": output_accuracy,
              "fail-details": fail_details}
    return result


def evaluate_fundaments(model_name, require_flat=False, use_extended=False):
    x, y = myds.load_all(require_flat, use_extended)
    return __evaluate(model_name, x, y)
        

def evaluate_ensemble(model_name):
    x, y = get_ensemble_dataset()
    return __evaluate(model_name, x, y)


if __name__ == "__main__":
    # res = evaluate_fundaments("lstm_09-24_15-59_0.9554", require_flat=False)
    res = evaluate_fundaments("dnn_09-26_10-31_1.0000",
                              require_flat=True, use_extended=True)
    print(res)
    with open("result_dnn.json", "w") as fp:
        json.dump(res, fp)
