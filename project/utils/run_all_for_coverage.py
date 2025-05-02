from pathlib import Path
from functools import partial
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.extmath import softmax
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann.estimation import Covariances
from moabb.datasets import BNCI2014_001, Zhou2016, BNCI2014_004, BNCI2014_002
from sklearn.utils import compute_sample_weight
from keras import callbacks, optimizers, utils

from project.utils.load_data import load_data
from project.utils.uncertainty_utils import find_best_temperature
from project.utils.rejection_coverage import accuracy_coverage_curve, get_uncertainty

from tensorflow.python.keras import backend as tfK
import keras_uncertainty.backend as KU           # <- existing module

# ------------------------------------------------------------------
# patch missing symbols only if they are absent
# ------------------------------------------------------------------
for _sym in ("mean", "square", "sum", "max", "min", "exp", "log"):
    if not hasattr(KU, _sym):
        setattr(KU, _sym, getattr(tfK, _sym))



_earlystop = callbacks.EarlyStopping(
    monitor="val_loss", patience=20, mode="min", restore_best_weights=True
)


def _train_single_scn(X_tr, y_tr, *, chans, samples, n_classes):
    from project.models.shallowConvNet.standard.standard_SCN_model import ShallowConvNet

    net = ShallowConvNet(nb_classes=n_classes,
                         Chans=chans,
                         Samples=samples,
                         dropoutRate=0.5)
    net.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                loss="categorical_crossentropy",
                metrics=["accuracy"])
    net.fit(X_tr, y_tr,
            epochs=100,
            batch_size=64,
            validation_split=0.1,
            callbacks=[_earlystop],
            verbose=0)
    return net


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


DATASETS = [
    ("BNCI 2014-002 (Steriade)", BNCI2014_002(), 2),
    ("Zhou 2016",               Zhou2016(),      3),
    ("BNCI 2014-004 (BCIC IV-2b)", BNCI2014_004(), 2),
    ("BNCI 2014-001 (BCIC IV-2a)", BNCI2014_001(), 4),
]


def _run_csplda(dataset, n_classes, temperature_scaling=False, *, random_state=42):
    from mne.decoding import CSP

    all_preds, all_labels, all_uncert = [], [], []

    for subject_id in range(1, len(dataset.subject_list) + 1):
        X, y, _ = load_data(dataset, subject_id, n_classes)
        y = LabelEncoder().fit_transform(y)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        pipe = make_pipeline(CSP(n_components=8), LDA())
        pipe.fit(X_tr, y_tr)

        if temperature_scaling:
            # find temperature on train set (distances are negative squared scores)
            distances_tr = -pipe.transform(X_tr) ** 2
            T = find_best_temperature(predictions=pipe.predict(X_tr),
                                      y_true=y_tr,
                                      distances=distances_tr)
            distances_te = -pipe.transform(X_te) ** 2
            proba = softmax(distances_te / T)
        else:
            proba = pipe.predict_proba(X_te)

        all_preds.append(proba)
        all_labels.append(y_te)
        all_uncert.append(get_uncertainty(proba, mode="uncertainty"))

    preds_cat  = np.concatenate(all_preds,   axis=0)
    labels_cat = np.concatenate(all_labels,  axis=0)
    uncert_cat = np.concatenate(all_uncert,  axis=0)
    return preds_cat, labels_cat, uncert_cat


def _run_mdrm(dataset, n_classes, temperature_scaling=False, *, random_state=42):
    from project.models.Riemann.MDRM_model import MDM

    all_preds, all_labels, all_uncert = [], [], []
    cov_estimator = Covariances(estimator='lwf')

    for subject_id in range(1, len(dataset.subject_list) + 1):
        X, y, _ = load_data(dataset, subject_id, n_classes)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        cov_estimator = Covariances(estimator='lwf')
        X_cov = cov_estimator.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_cov, y, test_size=0.2, random_state=42)
        weights = compute_sample_weight('balanced', y=y_train)

        model = MDM(metric=dict(mean='riemann', distance='riemann'))
        model.fit(X_train, y_train, sample_weight=weights)

        y_pred = model.predict(X_test)

        # Determine the confidence of the model
        if temperature_scaling:
            y_pred_train = model.predict(X_train)
            distances_train = -model.transform(X_train) ** 2
            temperature = find_best_temperature(y_pred_train, y_train, distances_train)

            distance_pred = model.transform(X_test)
            distances = -distance_pred ** 2
            prediction_proba = softmax(distances / temperature)
        else:
            prediction_proba = model.predict_proba(X_test)

        all_preds.append(prediction_proba)
        all_labels.append(y_test)
        all_uncert.append(get_uncertainty(prediction_proba, mode="uncertainty"))

    return (
        np.concatenate(all_preds,   axis=0),
        np.concatenate(all_labels,  axis=0),
        np.concatenate(all_uncert,  axis=0)
    )

def _run_scn_ensemble(dataset, n_classes,
                      n_members: int = 5,
                      *, random_state=42):

    chans_by_ds   = {2:15, 3:14, 4:3, 1:22}
    samples_by_ds = {2:2561, 3:1251, 4:1126, 1:1001}

    all_preds, all_labels, all_uncert = [], [], []

    for sid in range(1, len(dataset.subject_list) + 1):
        X, y, _ = load_data(dataset, sid, n_classes)
        y = utils.to_categorical(LabelEncoder().fit_transform(y), n_classes)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        member_softmax = []
        for _ in range(n_members):
            net = _train_single_scn(X_tr, y_tr,
                                    chans=chans_by_ds[n_classes],
                                    samples=samples_by_ds[n_classes],
                                    n_classes=n_classes)
            member_softmax.append(net.predict(X_te))

        proba = np.mean(member_softmax, axis=0)          # ensemble mean
        all_preds.append(proba)
        all_labels.append(y_te.argmax(1))
        all_uncert.append(get_uncertainty(proba))

    return (np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_uncert))


def _run_duq(dataset, n_classes, *, random_state=42):
    # from project.models.shallowConvNet.DUQ.SCN_model_DUQ import ShallowConvNet
    from project.models.shallowConvNet.DUQ.SCN_DUQ_model_new import ShallowConvNet

    all_preds, all_labels, all_uncert = [], [], []

    chans_by_ds = {2: 15, 3: 14, 4: 3, 1: 22}
    samples_by_ds = {2: 2561, 3: 1251, 4: 1126, 1: 1001}

    for sid in range(1, len(dataset.subject_list) + 1):
        X, y, _ = load_data(dataset, sid, n_classes)
        y = utils.to_categorical(LabelEncoder().fit_transform(y), n_classes)

        # reshape to (N, Ch, T, 1) expected by SCN
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        model = ShallowConvNet(nb_classes=n_classes,
                             Chans=chans_by_ds[n_classes],
                             Samples=samples_by_ds[n_classes],
                             dropoutRate=0.5)

        optimizer = optimizers.Adam(learning_rate=0.01)  # standard 0.001
        model.compile(loss="binary_crossentropy",
                      optimizer=optimizer, metrics=["categorical_accuracy"])

        model.fit(X_tr, y_tr,
                epochs=100,
                batch_size=64,
                validation_split=0.1,
                callbacks=[_earlystop],
                verbose=0)

        dist_tr = model.predict(X_tr) ** 2
        T = find_best_temperature(dist_tr.argmax(1), y_tr.argmax(1), dist_tr)

        dist_te = model.predict(X_te) ** 2
        proba   = softmax(dist_te / T)          # shape (n_te, C)

        all_preds.append(proba)
        all_labels.append(y_te.argmax(1))
        all_uncert.append(get_uncertainty(proba))   # 1 − confidence

    return (np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_uncert))


MODELS = [
    ("DUQ", partial(_run_duq)),
    ("Standard-CNN", partial(_run_scn_ensemble, n_members=1)),
    ("CNN-Ensemble", partial(_run_scn_ensemble, n_members=5)),
    ("CSP-LDA",   partial(_run_csplda,   temperature_scaling=False)),
    ("MDRM",      partial(_run_mdrm,     temperature_scaling=False)),
    ("MDRM-T",    partial(_run_mdrm,     temperature_scaling=True)),
]




def main(n_steps: int = 50, out_dir: str = "./graphs/accuracy_coverage") -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"► Saving plots to  {out_path.absolute()}")

    # storage: dict[dataset_idx][model_name] = (coverage, accuracy)
    curves = {i: {} for i, _ in enumerate(DATASETS)}

    for ds_idx, (ds_name, ds_obj, n_cls) in enumerate(DATASETS):
        # print(f"\n=== Dataset {ds_idx+1} / {len(DATASETS)}  |  {ds_name} ===")
        for model_name, runner in MODELS:
            # print(f"  → {model_name} …", end="", flush=True)

            preds, labels, uncert = runner(ds_obj, n_cls)

            coverage, accuracy, _ = accuracy_coverage_curve(
                labels, preds, uncert, step=0.01  # 1 % steps
            )
            # accuracy = 1.0 - risk
            curves[ds_idx][model_name] = (coverage, accuracy)
            # print(" done.")

    np.savez(out_path / "all_accuracy_coverage_curves.npz", **{
        f"ds{ds_idx}_{model_name.replace(' ', '_')}": (cov, acc)
        for ds_idx in curves
        for model_name, (cov, acc) in curves[ds_idx].items()
    })
    print("✓ cached curves → all_accuracy_coverage_curves.npz")


    # ------------------ plot one figure per dataset ------------------------ #
    for ds_idx, (ds_name, _, _) in enumerate(DATASETS):
        plt.style.use('classic')
        plt.figure(figsize=(5, 4))
        for model_name, (cov, acc) in curves[ds_idx].items():
            rej = 1.0 - cov
            plt.plot(rej, acc, lw=3, label=model_name)

        plt.xlim(0.0, 1.05)
        plt.ylim(0.0, 1.05)
        plt.xlabel("Rejection rate", fontsize=20)
        # plt.gca().invert_xaxis()
        plt.ylabel("Accuracy", fontsize=20)
        # plt.title(f"{ds_name}: Accuracy–Coverage", fontsize=13)
        plt.legend(loc="lower left", fontsize=11)
        plt.tight_layout()

        # filename = out_path / f"dataset{ds_idx+1}_accuracy_coverage.png"
        # plt.savefig(filename, dpi=150)
        filename = out_path / f"dataset{ds_idx + 1}_accuracy_coverage.pdf"
        plt.savefig(filename, bbox_inches="tight")  # PDF is vector; dpi not needed

        plt.close()
        print(f"  ↳ saved {filename.name}")

    print("\n✓ All done.")


if __name__ == "__main__":
    main()
