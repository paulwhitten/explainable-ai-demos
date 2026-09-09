import argparse
import json
import numpy as np
from datetime import datetime
import math
from load_emnist_resnet_data import emnist_cap_letter_classes, emnist_lower_letter_classes, emnist_digit_classes, emnist_classes

count_class_pred = 0
low_pred_count = 0

#TODO fix
def calc_class_from_predictions(preds):
    global count_class_pred
    global low_pred_count
    count_class_pred += 1
    index = 0
    max_pred = 0.0
    max_index = 0
    for pred in preds:
        if pred > max_pred:
            max_pred = pred
            max_index = index
        index += 1
    one_hot_pred = np.zeros((index), np.float32)
    one_hot_pred[max_index] = max_pred
    #TODO: should there be a certainty threshold?
    #TODO: could we track the other "close" predictions?
    if 0.70 > max_pred:
        low_pred_count += 1
    return max_index, one_hot_pred.tolist()

def get_num_classes(preds):
    max_class = max(preds)
    min_class = min(preds)
    return max_class - min_class + 1

parser = argparse.ArgumentParser(description='trains transforms for a set')
parser.add_argument('-k', '--kb_folder', 
                    help='The knowledgebase input folder')
parser.add_argument('-d', '--digits', action='store_const',
                    const=True, default=False, 
                    help='Flag to use digits.')
parser.add_argument('-c', '--cap', action='store_const',
                    const=True, default=False, 
                    help='Flag to use capital letters.')
parser.add_argument('-l', '--lower', action='store_const',
                    const=True, default=False, 
                    help='Flag to use lowercase letters.')
parser.add_argument('-a', '--all', action='store_const',
                    const=True, default=False, 
                    help='Flag to use all classes.')
parser.add_argument('-e','--exclude', nargs='+', help='transform names to exclude', required=False)
parser.add_argument('-s', '--save', action='store_const',
                    const=True, default=False, 
                    help='Flag to  save output')

args = parser.parse_args()

exclude = []
c_classes = []

if args.exclude:
    for excluded_transform in args.exclude:
        exclude.append(excluded_transform)

print("Excluded transforms:", exclude)

if args.digits:
    c_classes = emnist_digit_classes
elif args.cap:
    c_classes = emnist_cap_letter_classes
elif args.lower:
    c_classes = emnist_lower_letter_classes
elif args.all:
    c_classes = emnist_classes

print("classes:", c_classes)

batches = [
    "raw", "crossing", "endpoint", "fill", "skel-fill", "skel", "thresh", "line",
    "ellipse", "circle", "ellipse-circle", "chull"
]

confusion_matrix = np.zeros((len(c_classes), len(c_classes)), dtype=int)

stats_json = {}     # statistics on the training set
output_json = {}    # training data
test_data_json = {} # test data
output_json["propertyDescriptions"] = batches
stats_json["propertyDescriptions"] = batches
names = []
test_names = []
# overall accuracy stats for transforms index on each transform
probability_predictions = {}
probability_predictions["correctCounts"] = []
# prediction of all classes
training_predictions = []
test_predictions = []
# the predicted class number
training_pred_class = []
test_pred_class = []
# each image prediction data
#training_prediction_probabilties = []
#test_prediction_probabilties = []
labels = []
test_labels = []
test_name_lookup = {}
train_name_lookup = {}
counts = None
# list of per tranform results to use to determine training accuracry
training_accuracy = []

test_batch_index = 0
for batch_index, batch in enumerate(batches):
    batch_start_time = datetime.now()
    print("batch:", batch, batch_index+1, "of", len(batches))
    train_name_lookup[batch] = batch_index
    # load the files
    with open(args.kb_folder + "/" + batch + ".train.json", 'r') as infile:
        train_predictions = json.load(infile)
    with open(args.kb_folder + "/" + batch + ".test.json", 'r') as infile:
        test_prediction_list = json.load(infile)
    with open(args.kb_folder + "/" + batch + ".train-labels.json", 'r') as infile:
        train_label_list = json.load(infile)
    with open(args.kb_folder + "/" + batch + ".test-labels.json", 'r') as infile:
        test_label_list = json.load(infile)
    if batch_index == 0:
        # set up the array of predictions and probabilities
        # we need an array for each sample in the
        # training set.  That array will contain
        # the probabilities and classifications
        # for each transform
        for _ in range(len(train_predictions)):
            training_predictions.append([])
            #training_prediction_probabilties.append([])
            training_pred_class.append([])
        # populate the test prediction probabilities
        for _ in range(len(test_label_list)):
            #test_prediction_probabilties.append([])
            test_predictions.append([])
            test_pred_class.append([])
        labels = train_label_list
        counts = np.zeros((max(train_label_list)+1), np.uint32)
        for label in train_label_list:
            names.append(str(label) + "-" + str(counts[label]))
            counts[label] += 1
        probability_predictions["counts"] = counts.tolist()
        test_labels = test_label_list
    correct_counts = np.zeros((max(train_label_list)+1), np.uint32)
    print("len(train_predictions):", str(len(train_predictions)))
    batch_accuracy = []
    classes = get_num_classes(train_predictions)
    for pred_index, pred in enumerate(train_predictions):
        class_number = pred #, classes = calc_class_from_predictions(pred)
        training_predictions[pred_index].append(class_number)
        training_pred_class[pred_index].append(class_number)
        #training_prediction_probabilties[pred_index].append(1.0)
        if train_label_list[pred_index] == class_number:
            correct_counts[class_number] += 1
        batch_accuracy.append(class_number)
    
    probability_predictions["correctCounts"].append(correct_counts.tolist())

    #############################################
    #process test data
    #if batch != "thresh":
    #if batch != "thresh" and batch != "raw":
    if batch not in exclude:
        training_accuracy.append(batch_accuracy)
        test_names.append(batch)
        print(batch, "test_prediction_list length:", len(test_prediction_list))
        test_pred_classes = get_num_classes(test_prediction_list)
        for test_image_index, pred in enumerate(test_prediction_list):
            #for p_ix, p in enumerate(pred):
            #    if math.isnan(p):
            #        pred[p_ix] = 0.0
            #print("index:", test_image_index)
            #test_prediction_probabilties[test_image_index].append(pred)
            test_predicted_class = pred #, test_pred_classes = calc_class_from_predictions(pred)
            test_predictions[test_image_index].append(test_predicted_class)
            test_pred_class[test_image_index].append(test_predicted_class)
        test_batch_index += 1
    batch_end_time = datetime.now()
    print(batch, "done in:", batch_end_time - batch_start_time)

print("Transforms used:", test_names)

# figure out the probability each transform has in being correct for each class
print("calculating accuracy per transform")
header_1 = '| Character | '
header_2 = '| :----: | '
for i in range(len(batches)):
    header_1 += batches[i] + ' | '
    header_2 += ' :----: | '
print(header_1)
print(header_2)
table_lines = []
for i in range(len(c_classes)):
    table_lines.append('| ' + c_classes[i] + ' | ')
accuracy = []
#print('len probability_predictions["correctCounts"]:', len(probability_predictions["correctCounts"]))
for preds in probability_predictions["correctCounts"]: # loop on each transform
    transform_acc = []
    #print("preds length:", len(preds))
    for pred_index, pred in enumerate(preds):
        acc = 0.0
        if probability_predictions["counts"][pred_index] > 0:
            acc = pred / probability_predictions["counts"][pred_index]
        transform_acc.append(acc)
        table_lines[pred_index] += ' ' + f'{acc:.3}' + ' | '
    accuracy.append(transform_acc)
for line in table_lines:
    print(line)
print("accuracy complete")

# determine training accuracy
# Treating all transforms as equal
training_correct = 0
for label_ix, label in enumerate(labels):
    preds = np.zeros((max(train_label_list)+1), np.float32)
    for trans_preds in training_accuracy:
        preds[trans_preds[label_ix]] += 1.0
    pred = np.argmax(preds, axis=-1)
    if pred == label:
        training_correct += 1
print("Training accuracy:", training_correct / len(labels))

# figure out the test results
class_corrects = [0] * len(c_classes)
class_counts = [0] * len(c_classes)
test_correct = 0
test_scores = []
test_classifications = []
num_classes = get_num_classes(test_labels)
for test_index, label in enumerate(test_labels):
    class_counts[label] += 1
    scores = np.zeros((num_classes), np.float32)
    for transform_index, pred_class in enumerate(test_pred_class[test_index]):
        transform_name = test_names[transform_index]
        train_transform_num = train_name_lookup[transform_name]
        scores[pred_class] += accuracy[train_transform_num][pred_class]
    max_score = 0
    max_score_ix = 0
    for score_ix, score in enumerate(scores):
        if score > max_score:
            max_score = score
            max_score_ix = score_ix
    test_classifications.append(max_score_ix)
    test_scores.append(scores.tolist())
    if label == max_score_ix:
        test_correct += 1
        class_corrects[label] += 1
    confusion_matrix[label, max_score_ix] += 1

print('Test result accuracy per class:')
print('| Char | Accuracy | Correct | Total |')
print('| :----: | :----: | :----: | :----: |')
for i in range(len(c_classes)):
    if class_counts[i] > 0:
        print('| ' + c_classes[i] + ' | ' + f'{class_corrects[i]/class_counts[i]:.3}' +
          ' | ' + str(class_corrects[i]) + ' | ' + str(class_counts[i]) + ' | ')
print("Test accuracy:", test_correct, test_correct / len(test_labels))

header = " ,"
for c in c_classes:
    header += c + ","
print(header)
for row_letter, row in enumerate(confusion_matrix):
    row_data = c_classes[row_letter] + ","
    for col_letter, col_val in enumerate(row):
        row_data += str(col_val) + ","
    print(row_data)

output_json["predictions"] = training_predictions
#output_json["predictionProbabilities"] = training_prediction_probabilties
output_json["names"] = names
output_json["probabilityPredictions"] = probability_predictions
stats_json["probabilityPredictions"] = probability_predictions
output_json["accuracy"] = accuracy
stats_json["accuracy"] = accuracy
output_json["labels"] = labels
output_json["nameLookup"] = train_name_lookup

test_data_json["names"] = test_names
test_data_json["classNames"] = c_classes
test_data_json["labels"] = test_labels
#test_data_json["predictionProbabilities"] = test_prediction_probabilties
test_data_json["predictions"] = test_predictions
test_data_json["scores"] = test_scores
test_data_json["classifications"] = test_classifications
test_data_json["nameLookup"] = test_name_lookup

#print("low_pred_count:", low_pred_count, "total_pred:", count_class_pred, "=", low_pred_count / count_class_pred)

if args.save:
    print("saving stats...")
    with open(args.kb_folder + "/all-stats.json", 'w') as outfile:
        json.dump(stats_json, outfile)
    print("done.")

    print("saving test results...")
    with open(args.kb_folder + "/all-test-stats.json", 'w') as outfile:
        json.dump(test_data_json, outfile, indent=4)
    print("done.")

    print("saving all...")
    with open(args.kb_folder + "/all_with_names.json", 'w') as outfile:
        json.dump(output_json, outfile)
    print("done.")
