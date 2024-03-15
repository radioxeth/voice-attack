import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# imputer = SimpleImputer(strategy="mean")


from util import (
    CLASS_RECORDED,
    CLASS_TTS,
    mfccs_class_ids_from_files,
    shuffle_and_split,
)

# load mfccs
# model = GaussianNB()
# mfccs = np.load("mfccs.npy")


def pad_mfccs(mfccs):
    max_length = max(mfcc.shape[1] for mfcc in mfccs)
    # Pad MFCCs to have the same shape
    mfccs_padded = [
        np.pad(
            mfcc,
            ((0, 0), (0, max_length - mfcc.shape[1])),
            mode="constant",
            constant_values=0,
        )
        for mfcc in mfccs
    ]

    return mfccs_padded


# def pca_mfccs(mfccs):
#     mfccs = np.array(mfccs)
#     mfccs = mfccs.reshape(mfccs.shape[0], -1)
#     print(np.shape(mfccs))

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(mfccs)
#     pca = PCA(n_components=40)
#     X_pca = pca.fit_transform(X_train)
#     return X_pca


def pca_mfccs(mfccs):
    mfccs = np.array(mfccs)
    mfccs = mfccs.reshape(mfccs.shape[0], -1)

    # Initialize and apply imputation
    imputer = SimpleImputer(strategy="mean")
    mfccs_imputed = imputer.fit_transform(mfccs)

    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(mfccs_imputed)

    pca = PCA(n_components=40)
    X_pca = pca.fit_transform(X_train)
    return X_pca


def naive_bayes(recorded_directory_path, generated_directory_path):
    # open .npy files and get the mfcc
    mfccs_train = []
    class_ids_train = []
    mfccs_test = []
    class_ids_test = []

    recorded_entries = os.listdir(recorded_directory_path)
    generated_entries = os.listdir(generated_directory_path)

    recorded_files = [
        entry
        for entry in recorded_entries
        if os.path.isfile(os.path.join(recorded_directory_path, entry))
        and ".npy" in entry
    ]

    generated_files = [
        entry
        for entry in generated_entries
        if os.path.isfile(os.path.join(generated_directory_path, entry))
        and ".npy" in entry
    ]

    # shuffle and split the files
    recorded_files_train, recorded_files_test = shuffle_and_split(recorded_files, 0.9)
    generated_files_train, generated_files_test = shuffle_and_split(
        generated_files, 0.9
    )

    # open .npy files and get the mfcc
    mfccs_train, class_ids_train = mfccs_class_ids_from_files(
        recorded_directory_path,
        recorded_files_train,
        mfccs_train,
        class_ids_train,
        CLASS_RECORDED,
    )

    mfccs_train, class_ids_train = mfccs_class_ids_from_files(
        generated_directory_path,
        generated_files_train,
        mfccs_train,
        class_ids_train,
        CLASS_TTS,
    )

    # open .npy files and get the mfcc
    mfccs_test, class_ids_test = mfccs_class_ids_from_files(
        recorded_directory_path,
        recorded_files_test,
        mfccs_test,
        class_ids_test,
        CLASS_RECORDED,
    )

    mfccs_test, class_ids_test = mfccs_class_ids_from_files(
        generated_directory_path,
        generated_files_test,
        mfccs_test,
        class_ids_test,
        CLASS_TTS,
    )

    mfccs_train = pad_mfccs(mfccs_train)
    mfccs_train = pca_mfccs(mfccs_train)

    mfccs_test = pad_mfccs(mfccs_test)
    mfccs_test = pca_mfccs(mfccs_test)

    # # train the model
    model = GaussianNB()
    model.fit(mfccs_train, class_ids_train)

    # # test the model
    class_ids_pred = model.predict(mfccs_test)

    # # compute the accuracy
    accuracy = accuracy_score(class_ids_test, class_ids_pred)
    print(f"Accuracy: {accuracy}")

    # compute the confusion matrix
    cm = confusion_matrix(class_ids_test, class_ids_pred)

    # print the confusion matrix pretty
    print("Confusion Matrix")
    print(cm)

    # compute the centers
    # centers = np.array(
    #     [np.mean(mfccs_train[class_ids_train == i], axis=0) for i in range(2)]
    # )
    # return centers, mfccs_train, class_ids_train, mfccs_test, class_ids_test


def main():
    parser = argparse.ArgumentParser(
        description="Naive Bayes classifier for recorded and generated MFCCs"
    )
    parser.add_argument(
        "recorded_directory_path",
        type=str,
        help="The directory path to the recorded audio files",
    )
    parser.add_argument(
        "generated_directory_path",
        type=str,
        help="The directory path to the generated audio files",
    )
    args = parser.parse_args()
    naive_bayes(args.recorded_directory_path, args.generated_directory_path)
    results = []


if __name__ == "__main__":
    print("__main__")
    main()