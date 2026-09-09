#!/usr/bin/env python

import os
import json
import argparse
import subprocess

def create_dir(path):
    if os.path.isdir(path):
        print(f"Directory '{path}' exists.")
        return False
    else:
        os.makedirs(path, exist_ok=True)
        print(f"Created directory '{path}'.")
        return True

TRAIN = "train"
TEST = "test"
NAME = "name"
DATASET_TYPE = "dataset_type"
TRANSFORM_FOLDER = "transform_folder"
MODELS_FOLDER = "models_folder"
LOG_FOLDER ="log_folder"
KNOWLEDGEBASE_FOLDER = "knowledgebase_folder"
TRAINING_IMAGES_FILE = "training_images_file"
TRAINING_LABELS_FILE = "training_labels_file"
TEST_IMAGES_FILE = "test_images_file"
TEST_LABELS_FILE = "test_labels_file"
IMAGE_FILE_ORDERING = "image_file_ordering"
MODEL_TYPE = "model_type"
UNEXPLAINABLE_COMPONENT = "unexplaianble_component"
SAVE_IMAGES = "save_images"
SAVE_IMAGES_COUNT = "save_images_count"
SAVE_IMAGES_FOLDER = "save_images_folder"
THRESHOLD = "threshold"

TRANSFORM_LOGFILE = "transform.log.txt"
TRAIN_LOGFILE = "train.log.txt"
BUILD_KB_LOGFILE = "build_kb.log.txt"
PROCESS_KB_LOGFILE = "process_kb.log.txt"


# This function will prepare to save image data for transfomr
def prep_for_image_save(transform_command_array, config, data_type):
    if config[SAVE_IMAGES] and config[SAVE_IMAGES_COUNT] > 0 and config[SAVE_IMAGES_FOLDER]:
        save_img_folder = os.path.join(config[SAVE_IMAGES_FOLDER], config[DATASET_TYPE], config[NAME], data_type)
        create_dir(save_img_folder)
        process = subprocess.Popen(["./mk_image_folders.py", "-f", save_img_folder], stdout=log_file, stderr=subprocess.STDOUT)
        transform_command_array.append("--saveImages")
        transform_command_array.append("--saveLocation")
        transform_command_array.append(save_img_folder)
        process.wait()

def reorder_images(transform_command_array, config):
    if config[IMAGE_FILE_ORDERING] == "column":
        transform_command_array.append("--column")

# {
#     "name": "some_name",
#     "dataset_type": "mnist",
#     "transform_folder": "transforms",
#     "models_folder": "models",
#     "log_folder": "log",
#     "knowledgebase_folder": "kb",
#     "training_images_file": "",
#     "training_labels_file": "",
#     "test_images_file": "",
#     "test_labels_file": "",
#     "model_type": "cnn",
#     "unexplaianble_component": true,
#     "threshold": 0.05
# }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='runs the property based pipeline')
    parser.add_argument('-c', '--config', 
                        help='The pipeline configuration file')
    args = parser.parse_args()

    # read input file
    with open(args.config, 'r') as file:
        config = json.load(file)
    print("Configuration:", config)

    # create the log folder
    log_folder = os.path.join(config[LOG_FOLDER], config[DATASET_TYPE], config[MODEL_TYPE], config[NAME])
    create_dir(log_folder)

    # transform
    train_transform_folder = os.path.join(config[TRANSFORM_FOLDER], config[DATASET_TYPE], TRAIN, config[NAME])
    test_transform_folder = os.path.join(config[TRANSFORM_FOLDER], config[DATASET_TYPE], TEST, config[NAME])
    if create_dir(train_transform_folder) and create_dir(test_transform_folder):
        print("Transforming  training:")
        with open(os.path.join(log_folder, TRANSFORM_LOGFILE), "w") as log_file:
            transform_command = ['./transform_parallel.py', '-i', config[TRAINING_IMAGES_FILE], '-l', config[TRAINING_LABELS_FILE], '-o', train_transform_folder]
            prep_for_image_save(transform_command, config, TRAIN)
            reorder_images(transform_command, config)
            process = subprocess.Popen(transform_command, stdout=log_file, stderr=subprocess.STDOUT)
            process.wait()
            print("Train transform result:", process.returncode)
            transform_command = ['./transform_parallel.py', '-i', config[TEST_IMAGES_FILE], '-l', config[TEST_LABELS_FILE], '-o', test_transform_folder]
            prep_for_image_save(transform_command, config, TEST)
            reorder_images(transform_command, config)
            process = subprocess.Popen(transform_command, stdout=log_file, stderr=subprocess.STDOUT)
            process.wait()
            print("Test transform result:", process.returncode)

    # train models
    model_folder = os.path.join(config[MODELS_FOLDER], config[DATASET_TYPE], config[MODEL_TYPE], config[NAME])
    train_command = f"./train_{config[MODEL_TYPE]}_transforms.py"
    if create_dir(model_folder):
        print(f"Training {config[MODEL_TYPE]} models.")
        with open(os.path.join(log_folder, TRAIN_LOGFILE), "w") as log_file:
            process = subprocess.Popen([train_command, "-r", train_transform_folder, "-e", test_transform_folder, "-o", model_folder], stdout=log_file, stderr=subprocess.STDOUT)
            process.wait()

    # build knowledgebase
    build_kb_command = "./build_kb"
    if config[MODEL_TYPE] == "cnn":
        build_kb_command += "_cnn"
    elif config[MODEL_TYPE] == "resnet50":
        build_kb_command += "_resnet50"
    else:
        print("Using base build kb script")
    build_kb_command += ".py"
    kb_folder = os.path.join(config[KNOWLEDGEBASE_FOLDER], config[DATASET_TYPE], config[MODEL_TYPE], config[NAME])
    if create_dir(kb_folder):
        print("Building knowledgebase.")
        with open(os.path.join(log_folder, BUILD_KB_LOGFILE), "w") as log_file:
            process = subprocess.Popen([build_kb_command, "-r", train_transform_folder, "-e", test_transform_folder, "-m", model_folder, "-o", kb_folder], stdout=log_file, stderr=subprocess.STDOUT)
            process.wait()

    # process knowledgebase
    logfile = os.path.join(log_folder, PROCESS_KB_LOGFILE)
    with open(os.path.join(log_folder, PROCESS_KB_LOGFILE), "w") as log_file:
        process = subprocess.Popen(["./process_kb_conf_matrix.py", "-k", kb_folder], stdout=log_file, stderr=subprocess.STDOUT)
        process.wait()
        js_process_kb_command = ["node", "process_kb.js", "-k", kb_folder, "-e", "product", "-t", str(config[THRESHOLD])]
        if config[UNEXPLAINABLE_COMPONENT]:
            js_process_kb_command.append("-u")
        process = subprocess.Popen(js_process_kb_command, stdout=log_file, stderr=subprocess.STDOUT)
        process.wait()


