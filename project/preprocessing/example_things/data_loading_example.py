import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

import moabb
from moabb.datasets import BNCI2014_001, utils
from moabb.evaluations import CrossSessionEvaluation
from moabb.paradigms import LeftRightImagery
from moabb.pipelines.features import LogVariance

moabb.set_log_level("info")

pipelines = {}
pipelines["AM+LDA"] = make_pipeline(LogVariance(), LDA())
parameters = {"C": np.logspace(-2, 2, 10)}
clf = GridSearchCV(SVC(kernel="linear"), parameters)
pipe = make_pipeline(LogVariance(), clf)

pipelines["AM+SVM"] = pipe

# print(LeftRightImagery().datasets)
# print(utils.dataset_search(paradigm="imagery", min_subjects=6))

dataset = BNCI2014_001()
dataset.subject_list = dataset.subject_list[:2]
#datasets = [dataset]

print(dataset.get_data())
exit()

fmin = 8
fmax = 35
paradigm = LeftRightImagery(fmin=fmin, fmax=fmax)

evaluation = CrossSessionEvaluation(
    paradigm=paradigm, datasets=datasets, suffix="examples", overwrite=False
)
results = evaluation.process(pipelines)

print(results.head())