import tensorflow as tf
import os

from utils import data_generator

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
SEED = 42
MODEL_DIR = "model_artifacts"  # Directory to store models
