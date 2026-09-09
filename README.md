# Source Listing for An AI Architecture with the Capability to Explain Recognition Results

This repository contains a set of programs implementing the property-based explainable architecture for the paper 
[An AI Architecture with the Capability to Explain Recognition Results](https://doi.org/10.1109/ICMI60790.2024.10586116)

## Prerequisites

MNIST and EMNIST data

Python 3


## New `pipeline` Process

1. Edit or create a `pipeline*.json` file with the appropriate pipeline.
1. Run `./pipeline.py -c new_config_json`


### Pipeline Config

The following section details the JSON pipeline config.

Here is an example of a config.

```JSON
{
    "name": "some_name",
    "dataset_type": "mnist",
    "transform_folder": "transforms",
    "models_folder": "models",
    "log_folder": "log",
    "knowledgebase_folder": "kb",
    "training_images_file": "mnist/download/train-images-idx3-ubyte",
    "training_labels_file": "mnist/download/train-labels-idx1-ubyte",
    "test_images_file": "mnist/download/t10k-images-idx3-ubyte",
    "test_labels_file": "mnist/download/t10k-labels-idx1-ubyte",
    "model_type": "cnn",
    "unexplaianble_component": true,
    "threshold": 0.05
}
```

Subfolders are created according to the following JSON configuration fields fields:

`dataset_type/model_type/name`

The field value descriptions are:

- `name` - The name of the pipeline
- `dataset_type` - Field to differentiate between MNIST and EMNIST datasets
- `transform_folder` - the base folder for where transforms go
- `models_folder` - The base folder where trained models go
- `log_folder` - The base folder where the logs go
- `knowledgebase_folder` - The base folder of where the knowledgebase will go
- `training_images_file` - The training images file
- `training_labels_file` - The training labels file
- `test_images_file` - The test images file
- `test_labels_file` - The test labels file
- `model_type` - The type of model.  Potential values are `nn`, `svm`, `cnn`, and `resnet50`
- `unexplaianble_component` - Boolean indicating if an unexplainable component should be used
- `threshold` - A number indicating the threshold value for considering votes


## Old Process

1. Create a folder for transformed data
1. Transform MNIST and EMNIST data using the program `transform_parallel.py` into the folder for transforms
1. Create a folder to store the ML models
1. Train models using the program `train_svm_transforms.py` (support vector machines) or `train_nn_transforms.py` (multi-layer perceptrons) storing in the appropriate folder
1. Create a kb folder.
1. Build the knowledgebase (kb) by running the program `build_kb.py` with output of the kb folder
1. Run copy <model_foler>/*.json <kb_folder>/
1. Run the program `process_kb_conf_matrix.py` on the kb folder to generate confusion matrices and metrics
1. Run the program `node process_kb.js` against the kb folder to get results against test data with various effectiveness metrics and compare mixed versus only explainable via command line options.

## Web api

### prerequisites

```sh
pip install imageio
pip install fastapi
pip install uvicorn
```

A web api is implemented in `main_api.py` using fast api.

Start by opening the Python virtual environment and then running:

```sh
uvicorn main_api:app --reload
```

The API will serve:

- [An http web interface at http://127.0.0.1:8000/static/web_draw.html](http://127.0.0.1:8000/static/web_draw.html)
- POST to http://127.0.0.1:8000//submit that the page submits JSON data to for recognition from the webpage
