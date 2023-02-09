
from imagenet_x import get_factor_accuracies, error_ratio
from imagenet_x import plots


factor_accs = get_factor_accuracies("model_predictions/base/")
error_ratios = error_ratio(factor_accs)

print(error_ratios)

plots.model_comparison(factor_accs.reset_index(), fname="figures/error_ratios.pdf")
