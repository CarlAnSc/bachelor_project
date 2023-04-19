import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.imagenet_x import get_factor_accuracies, error_ratio
from src.imagenet_x.utils import load_model_predictions, get_annotation_path
from src.imagenet_x import plots

models, top_1_accs = load_model_predictions('predictionsplot')
factor_accs = get_factor_accuracies("predictionsplot/")
error_ratio = error_ratio(factor_accs)
plots.set_color_palette()
plt = plots.model_comparison(factor_accs.reset_index(), fname='error_ratios_on_imagenet_x.png')