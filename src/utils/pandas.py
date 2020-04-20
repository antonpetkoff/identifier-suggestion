import pandas as pd
from itertools import chain


def lists_to_series(lists):
    return pd.Series(chain.from_iterable(lists))
