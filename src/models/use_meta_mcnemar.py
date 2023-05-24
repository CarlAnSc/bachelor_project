import numpy as np
import pandas as pd
import sklearn
from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar
import scipy.stats as st

def mcnemar_ML(Matrix, alpha=0.05):
    """Code from the toolBox from course 02450 (Introduction to Machine Learning and Data mining
    However, it has been modified to fit the purpose of this project."""

    # perform McNemars test

    n = sum(Matrix.flat)
    n12 = Matrix[0, 1]
    n21 = Matrix[1, 0]

    # thetahat = (n12-n21)/n
    thetahat = (n21 - n12) / n

    Etheta = thetahat

    Q = (
        n**2
        * (n + 1)
        * (Etheta + 1)
        * (1 - Etheta)
        / ((n * (n12 + n21) - (n12 - n21) ** 2))
    )
    # Q = n**2 * (n+1) * (Etheta+1) * (1-Etheta) / ( (n*(n12+n21) - (n21-n12)**2) )

    p = (Etheta + 1) * 0.5 * (Q - 1)
    # p = (1-Etheta)*0.5 * (Q-1)
    q = (1 - Etheta) * 0.5 * (Q - 1)
    # q = (Etheta + 1)*0.5 * (Q-1)

    CI = tuple(lm * 2 - 1 for lm in st.beta.interval(1 - alpha, a=p, b=q))

    p = 2 * st.binom.cdf(min([n12, n21]), n=n12 + n21, p=0.5)
    print("Result of McNemars test using alpha=", alpha)
    print("Comparison matrix n")
    print(Matrix)
    if n12 + n21 <= 10:
        print("Warning, n12+n21 is low: n12+n21=", (n12 + n21))

    print(f"Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] ={CI}")
    print(
        f"p-value for two-sided test A and B have same accuracy (exact binomial test): p={p}"
    )

    return thetahat, CI, p


def main():
    df = pd.read_csv("data/resnet_preds.csv")

    acc_base = sklearn.metrics.accuracy_score(df["truth"].values, df["base"].values)
    acc_trained = sklearn.metrics.accuracy_score(
        df["truth"].values, df["trained"].values
    )

    print(f"Base accuracy: {acc_base}")
    print(f"Trained accuracy: {acc_trained}")

    M_table = mcnemar_table(
        y_target=df["truth"].values,
        y_model1=df["base"].values,
        y_model2=df["trained"].values,
    )

    mcnemar_ML(M_table)


if __name__ == "__main__":
    main()
