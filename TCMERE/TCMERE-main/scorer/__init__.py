from constants import *

from scorer.biorelex import evaluate_biorelex

def evaluate(model, dataset, type):
    if type == BIORELEX:
        return evaluate_biorelex(model, dataset)
