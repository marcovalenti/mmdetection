import pandas as pd
import numpy as np


def count_per_iou_range(min_range=0.05):
    def f(max_overlaps):
        df = pd.DataFrame({"iou": max_overlaps.tolist()})
        return df['iou'].value_counts(
            bins=np.arange(0, 1, min_range),
            sort=False
        )
    return f
