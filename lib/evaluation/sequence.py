from collections import Counter
from lib.preprocessing.tokens import get_subtokens

"""

compute_f1_score('transformSearchResponse', 'modifySearchResponseData')
> {'precision': 0.5, 'recall': 0.6666666666666666, 'f1': 0.5714285714285715}

"""
def compute_f1_score(target_token, predicted_token):
    target_subtokens = get_subtokens(target_token) # a.k.a required subtokens
    predicted_subtokens = get_subtokens(predicted_token)
    overlapping = Counter(target_subtokens) & Counter(predicted_subtokens)
    overlapping_count = sum(overlapping.values())

    precision = 1.0 * overlapping_count / len(predicted_subtokens)
    recall = 1.0 * overlapping_count / len(target_subtokens)
    f1 = (2.0 * precision * recall) / (precision + recall)

    return { 'precision': precision, 'recall': recall, 'f1': f1 }
