import numpy as np
import os

for file in os.listdir():
    if file.__contains__("baseline"):
        print(file)
        arr_base = np.loadtxt(file, delimiter=",")

        mean = np.mean(arr_base)
        var = np.var(arr_base)
        print(f"Mean is acc. is {mean}")
        print(f"Variance is {var}")

    if not file.__contains__("baseline") and not file.__contains__(".py"):
        print(file)
        arr_nx = np.loadtxt(file)
        print(f"ImageNet-X subset accuracy is {arr_nx}")
