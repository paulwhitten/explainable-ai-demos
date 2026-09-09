"""Case-based explanation service backed by the case-explainer package.

Exposes two retrieval spaces over the same MNIST MLP and training set:

- feature space (raw-pixel similarity), and
- hidden-layer activation space (the model's learned representation),

so a caller can compare the training cases each space retrieves.
"""

import os
import json
import pickle
import base64
from io import BytesIO

import numpy as np
import imageio
from sklearn.neural_network import MLPClassifier

from case_explainer import CaseExplainer, HiddenActivations

NN_MODEL_NAME = "nn.model"
TRAIN_LABEL_NAME = "train_labels.json"
TRAIN_DATA_NAME = "train_data.json"
TRAIN_DATA_NPY_NAME = "train_data.npy"


class _LabelMLPClassifier(MLPClassifier):
    """MLPClassifier whose predict returns integer class labels.

    The stored model was trained on one-hot targets, so stock predict returns a
    one-hot matrix; case-explainer expects scalar class labels.
    """

    def predict(self, X):
        out = np.asarray(super().predict(X))
        if out.ndim == 2 and out.shape[1] > 1:
            return np.argmax(out, axis=1)
        return out


def _load_train_data(folder):
    npy_path = os.path.join(folder, TRAIN_DATA_NPY_NAME)
    if os.path.exists(npy_path):
        return np.asarray(np.load(npy_path, mmap_mode='r'), dtype=np.float32)
    with open(os.path.join(folder, TRAIN_DATA_NAME)) as f:
        return np.asarray(json.load(f), dtype=np.float32)


def _neighbor_to_dict(neighbor):
    pixels = (np.asarray(neighbor.features) * 255).astype(np.uint8).reshape((28, 28))
    buf = BytesIO()
    imageio.imwrite(buf, pixels, format='png')
    b64_img = base64.b64encode(buf.getbuffer()).decode('utf-8')
    return {
        "index": int(neighbor.index),
        "label": int(neighbor.label),
        "distance": float(neighbor.distance),
        "b64_image": b64_img,
    }


class CaseExplainerService:
    """Feature-space and activation-space case explanations for MNIST."""

    def __init__(self, folder, k=10):
        self.k = k
        with open(os.path.join(folder, NN_MODEL_NAME), 'rb') as f:
            mlp = pickle.load(f)
        # reclass so predict returns labels while the MLP internals stay intact
        mlp.__class__ = _LabelMLPClassifier
        self.model = mlp

        with open(os.path.join(folder, TRAIN_LABEL_NAME)) as f:
            y_train = np.asarray(json.load(f))
        X_train = _load_train_data(folder)
        class_names = {i: str(i) for i in range(10)}

        # MNIST pixels are already on a common 0-1 scale, so disable standardizing.
        # Feature space stays pure (no model/similarity); the model is supplied at
        # explain time only to compute the prediction.
        self.feature_explainer = CaseExplainer(
            X_train=X_train, y_train=y_train, k=k,
            algorithm="brute", scale_data=False, class_names=class_names,
        )
        self.activation_explainer = CaseExplainer(
            X_train=X_train, y_train=y_train, k=k,
            algorithm="brute", scale_data=False, class_names=class_names,
            similarity=HiddenActivations(model=mlp),
        )

    def _explain(self, explainer, x_norm):
        explanation = explainer.explain_instance(test_sample=x_norm, model=self.model)
        return {
            "prediction": int(explanation.predicted_class),
            "correspondence": float(explanation.correspondence),
            "correspondence_interpretation": explanation.correspondence_interpretation,
            "neighbors": [_neighbor_to_dict(n) for n in explanation.neighbors],
        }

    def explain_feature(self, image_1d):
        """Feature-space explanation. image_1d holds 0-255 pixel values."""
        x = np.asarray(image_1d, dtype=np.float32) / 255.0
        return self._explain(self.feature_explainer, x)

    def explain_both(self, image_1d):
        """Feature- and activation-space explanations for the same input."""
        x = np.asarray(image_1d, dtype=np.float32) / 255.0
        feature = self._explain(self.feature_explainer, x)
        activation = self._explain(self.activation_explainer, x)
        return {
            "prediction": feature["prediction"],
            "feature": feature,
            "activation": activation,
        }
