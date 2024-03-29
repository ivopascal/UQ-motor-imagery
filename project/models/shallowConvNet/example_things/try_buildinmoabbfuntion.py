import keras.optimizers
import torch
from keras.backend import categorical_crossentropy
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from moabb import setup_seed, benchmark
from moabb.datasets import BNCI2014_001
from moabb.evaluations import CrossSessionEvaluation
from moabb.paradigms import MotorImagery
from moabb.pipelines.deep_learning import KerasShallowConvNet
from sklearn.pipeline import make_pipeline

from braindecode.models import EEGNetv4
from braindecode import EEGClassifier
from braindecode.models import EEGNetv4
from skorch.callbacks import EarlyStopping, EpochScoring
from skorch.dataset import ValidSplit


def main():
    seed = 42
    setup_seed(42)

    dataset = BNCI2014_001()
    events = ["right_hand", "left_hand"]
    paradigm = MotorImagery(
        events=events, n_classes=len(events), fmin=7.5, fmax=30, tmin=0, tmax=None
    )
    subjects = [1]
    X, _, _ = paradigm.get_data(dataset=dataset, subjects=subjects)

    # #optimizer = Adam(learning_rate=0.001)
    # clf = KerasShallowConvNet(
    #     # module=KerasShallowConvNet,
    #     optimizer= keras.optimizers.Adam,
    #     optimizer__lr=0.001,
    #     batch_size=64,
    #     loss=categorical_crossentropy,
    #     # callbacks=[
    #     #     EarlyStopping(monitor="valid_loss", patience=5),
    #     #     EpochScoring(
    #     #         scoring="accuracy", on_train=True, name="train_acc", lower_is_better=False
    #     #     ),
    #     #     EpochScoring(
    #     #         scoring="accuracy", on_train=False, name="valid_acc", lower_is_better=False
    #     #     ),
    #     # ],
    # )
    #
    # # Create the pipelines
    # pipes = {}
    # pipes["KerasShallowConvNet"] = make_pipeline(clf)

    # results = benchmark(
    #     pipelines=pipes[0],
    #     evaluations=["WithinSession"],
    #     paradigms=["LeftRightImagery"],
    #     include_datasets=datasets,
    #     results="./results/",
    #     overwrite=False,
    #     plot=False,
    #     output="./benchmark/",
    #     n_jobs=-1,
    # )

    # clf = KerasShallowConvNet(
    #     loss=categorical_crossentropy,
    #     optimizer=Adam,
    #     epochs=100,
    #     batch_size=64,
    #     validation_split=0.2
    # )

    clf = EEGClassifier(
        module=EEGNetv4,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.001,
        batch_size=64,
        max_epochs=100,
        train_split=ValidSplit(0.2, random_state=seed),
        callbacks=[
            EarlyStopping(monitor="valid_loss", patience=10),
            EpochScoring(
                scoring="accuracy", on_train=True, name="train_acc", lower_is_better=False
            ),
            EpochScoring(
                scoring="accuracy", on_train=False, name="valid_acc", lower_is_better=False
            ),
        ],
        verbose=1,  # Not printing the results for each epoch
    )

    # Create the pipelines
    pipes = {}
    pipes["KerasShallowConvNet"] = make_pipeline(clf)

    evaluation = CrossSessionEvaluation(
        paradigm=paradigm,
        datasets=dataset,
        suffix="braindecode_example",
        overwrite=True,
        return_epochs=True,
        n_jobs=1,
    )

    results = evaluation.process(pipes)
    print(results.head())


if __name__ == '__main__':
    main()