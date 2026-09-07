import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


def ef(y_true, y_score, percentage=10, higher_is_better=True):

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # sorting
    if higher_is_better:
        idx = np.argsort(-y_score)
    else:
        idx = np.argsort(y_score)

    y_true_sorted = y_true[idx]

    # top percentage
    top_k = int(len(y_true) * (percentage / 100.0))
    top_k = max(top_k, 1)

    # active counts
    actives_total = np.sum(y_true)
    actives_top = np.sum(y_true_sorted[:top_k])

    if actives_total == 0:
        return 0.0

    ef_value = (
        (actives_top / top_k)
        /
        (actives_total / len(y_true))
    )

    return ef_value


def calculate_efs(y, pred):

    ef05 = ef(
        y,
        pred,
        percentage=0.5
    )

    ef1 = ef(
        y,
        pred,
        percentage=1
    )

    ef5 = ef(
        y,
        pred,
        percentage=5
    )

    return ef05, ef1, ef5



def run(base):

    if '1zdk' in base:

        files = {

            # Internal ID: (name displayed in the output CSV, input file)
            "PRISeq": ("PRISeq", base + "priseq.txt"),

        }


    # read experiment
    exp_raw = pd.read_csv(
        base + "exp.txt",
        header=None
    ).iloc[:,0].values


    # top 1% active
    N = int(0.01 * len(exp_raw))

    threshold = np.sort(exp_raw)[N-1]

    y = (
        exp_raw <= threshold
    ).astype(int)


    print(
        "Method".ljust(22)
        +
        "EF0.5%".ljust(16)
        +
        "EF1%".ljust(16)
        +
        "EF5%".ljust(16)
        +
        "AUROC".ljust(16)
    )


    results=[]


    for method_id, (method_name, path) in files.items():


        # score direction correction

        if method_id in [

            'PRISeq',
        ]:

            pred = pd.read_csv(
                path,
                header=None
            ).iloc[:,0]



        # metrics

        auroc = roc_auc_score(
            y,
            pred
        )


        ef05,ef1,ef5 = calculate_efs(
            y,
            pred
        )


        print(
            f"{method_name:<22}"
            f"{ef05:<16.2f}"
            f"{ef1:<16.2f}"
            f"{ef5:<16.2f}"
            f"{auroc:<16.4f}"
        )


        results.append([

            method_name,

            f"{ef05:.2f}",
            f"{ef1:.2f}",
            f"{ef5:.2f}",
            f"{auroc:.2f}",

        ])



    # save csv

    df_results = pd.DataFrame(

        results,

        columns=[

            "Method",

            "EF0.5%",
            "EF1%",
            "EF5%",
            "AUROC",

        ]

    )


    output_csv = base[:-1]+"metric.csv"


    df_results.to_csv(
        output_csv,
        index=False
    )



run(base="1zdk/")

