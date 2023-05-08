import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
import pandas as pd


def get_mean_and_std(df):
    mean = df.mean(axis=0)
    df_results = df.append(mean, ignore_index=True)
    std = df.std(axis=0, numeric_only=True)
    df_results = df_results.append(std, ignore_index=True)
    df['seed'] = df['seed'].astype('str')
    df_results.loc[df_results.index[-2], ['seed']] = 'mean'
    df_results.loc[df_results.index[-1], ['seed']] = 'std'
    return df_results


def get_seconds_since_day(time_str):
    time = datetime.datetime.strptime(str(time_str), "%Y-%m-%dT%H:%M:%S.%f")
    return time.time().hour * 3600 + time.time().minute * 60 + time.time().second


def get_day_of_week(time_str):
    time = datetime.datetime.strptime(str(time_str), "%Y-%m-%dT%H:%M:%S.%f")
    return time.weekday()


def train_and_evaluate(X, y, input_seed, comb_model=None, vec_model=None, occ_model=None, ENSEMBLE=True,
                       oversampling_factor=1, xgb=False, epochs=200, batch_size=64):
    output = {}
    input_random_state = input_seed
    output['seed'] = str(input_seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=input_random_state, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15,
                                                          random_state=input_random_state, stratify=y_train)

    augment_oversampling = True
    if augment_oversampling:
        X_pos = X_train[np.where(y_train)[0], :]
        y_pos = y_train[np.where(y_train)[0]]
        for _ in range(oversampling_factor - 1):
            X_train = np.concatenate((X_train, X_pos), axis=0)
            y_train = np.concatenate((y_train, y_pos), axis=0)

    X_train_vec, X_train_occ = np.take(X_train, 0, axis=1), np.take(X_train, 1, axis=1)
    X_valid_vec, X_valid_occ = np.take(X_valid, 0, axis=1), np.take(X_valid, 1, axis=1)
    X_test_vec, X_test_occ = np.take(X_test, 0, axis=1), np.take(X_test, 1, axis=1)

    X_train_vec = np.array(list(X_train_vec), dtype=np.float32)
    X_valid_vec = np.array(list(X_valid_vec), dtype=np.float32)
    X_test_vec = np.array(list(X_test_vec), dtype=np.float32)

    X_train_occ = np.array(list(X_train_occ), dtype=np.float32)
    X_valid_occ = np.array(list(X_valid_occ), dtype=np.float32)
    X_test_occ = np.array(list(X_test_occ), dtype=np.float32)

    X_vec_scaler = StandardScaler()
    X_vec_scaler.fit(X_train_vec)
    X_train_vec = X_vec_scaler.transform(X_train_vec)
    X_valid_vec = X_vec_scaler.transform(X_valid_vec)
    X_test_vec = X_vec_scaler.transform(X_test_vec)

    if xgb:
        X_train_comb = np.append(X_train_vec, X_train_occ, axis=1)
        X_valid_comb = np.append(X_valid_vec, X_valid_occ, axis=1)
        X_test_comb = np.append(X_test_vec, X_test_occ, axis=1)

    tf.random.set_seed(input_seed)

    if comb_model:
        if xgb:
            comb_model.fit(X_train_comb, y_train)
            main_y_pred = comb_model.predict(X_test_comb)
        else:
            comb_model.fit(x=[X_train_occ, X_train_vec], y=y_train,
                           validation_data=([X_valid_occ, X_valid_vec], y_valid),
                           batch_size=batch_size, epochs=epochs, verbose=0)
            main_y_pred = comb_model.predict([X_test_occ, X_test_vec], verbose=0)

        main_y_pred = 1 * (main_y_pred > 0.5)
        main_acc = keras.metrics.BinaryAccuracy(name='accuracy')
        main_acc.reset_state()
        main_acc.update_state(main_y_pred, y_test)
        output['main_accuracy'] = main_acc.result().numpy()

        main_precision = keras.metrics.Precision(name='precision')
        main_precision.reset_state()
        main_precision.update_state(main_y_pred, y_test)
        output['main_precision'] = main_precision.result().numpy()

        main_recall = keras.metrics.Recall(name='recall')
        main_recall.reset_state()
        main_recall.update_state(main_y_pred, y_test)
        output['main_recall'] = main_recall.result().numpy()

        main_auc = keras.metrics.AUC(name='auc')
        main_auc.reset_state()
        main_auc.update_state(main_y_pred, y_test)
        output['main_auc'] = main_auc.result().numpy()

        output['main_f1'] = f1_score(main_y_pred, y_test)

    # Occupancy only
    if occ_model:
        if xgb:
            occ_model.fit(X_train_occ, y_train)
            occ_y_pred_raw = occ_model.predict(X_test_occ)
        else:
            occ_model.fit(x=X_train_occ, y=y_train,
                          validation_data=(X_valid_occ, y_valid),
                          batch_size=batch_size, epochs=epochs, verbose=0)
            occ_y_pred_raw = occ_model.predict(X_test_occ, verbose=0)
        occ_y_pred = 1 * (occ_y_pred_raw > 0.5)

        occ_acc = keras.metrics.BinaryAccuracy(name='accuracy')
        occ_acc.reset_state()
        occ_acc.update_state(occ_y_pred, y_test)
        output['occ_accuracy'] = occ_acc.result().numpy()

        occ_precision = keras.metrics.Precision(name='precision')
        occ_precision.reset_state()
        occ_precision.update_state(occ_y_pred, y_test)
        output['occ_precision'] = occ_precision.result().numpy()

        occ_recall = keras.metrics.Recall(name='recall')
        occ_recall.reset_state()
        occ_recall.update_state(occ_y_pred, y_test)
        output['occ_recall'] = occ_recall.result().numpy()

        occ_auc = keras.metrics.AUC(name='auc')
        occ_auc.reset_state()
        occ_auc.update_state(occ_y_pred, y_test)
        output['occ_auc'] = occ_auc.result().numpy()

        output['occ_f1'] = f1_score(occ_y_pred, y_test)

    # vec only
    if vec_model:
        if xgb:
            vec_model.fit(X_train_vec, y_train)
            tab_y_pred_raw = vec_model.predict(X_test_vec)

        else:
            vec_model.fit(x=X_train_vec, y=y_train,
                          validation_data=(X_valid_vec, y_valid),
                          batch_size=200, epochs=epochs, verbose=0)
            tab_y_pred_raw = vec_model.predict(X_test_vec, verbose=0)
        tab_y_pred = 1 * (tab_y_pred_raw > 0.5)

        tab_acc = keras.metrics.BinaryAccuracy(name='accuracy')
        tab_acc.reset_state()
        tab_acc.update_state(tab_y_pred, y_test)
        output['vec_only_accuracy'] = tab_acc.result().numpy()

        tab_precision = keras.metrics.Precision(name='precision')
        tab_precision.reset_state()
        tab_precision.update_state(tab_y_pred, y_test)
        output['vec_only_precision'] = tab_precision.result().numpy()

        tab_recall = keras.metrics.Recall(name='recall')
        tab_recall.reset_state()
        tab_recall.update_state(tab_y_pred, y_test)
        output['vec_only_recall'] = tab_recall.result().numpy()

        tab_auc = keras.metrics.AUC(name='auc')
        tab_auc.reset_state()
        tab_auc.update_state(tab_y_pred, y_test)
        output["vec_only_auc"] = tab_auc.result().numpy()

        output["vec_only_f1"] = f1_score(tab_y_pred, y_test)

    # find good a combination factor (a) for the ensemble
    if ENSEMBLE:

        a_list = np.arange(0, 1, 0.05)
        aucs = []
        if xgb:
            tab_y_pred_raw_valid = vec_model.predict(X_valid_vec)
            occ_only_y_pred_raw_valid = occ_model.predict(X_valid_occ)
        else:
            tab_y_pred_raw_valid = vec_model.predict(X_valid_vec, verbose=0)
            occ_only_y_pred_raw_valid = occ_model.predict(X_valid_occ, verbose=0)
        for a in a_list:
            pred = (tab_y_pred_raw_valid * a) + occ_only_y_pred_raw_valid * (1 - a)
            pred = pred > 0.5
            tab_auc = keras.metrics.AUC(name='auc')
            tab_auc.reset_state()
            tab_auc.update_state(pred, y_valid)
            aucs.append(tab_auc.result().numpy())

        best_a = a_list[np.argmax(aucs)]

        a_preds_raw = (tab_y_pred_raw * best_a) + (occ_y_pred_raw * (1 - best_a))
        a_preds = a_preds_raw > 0.5

        a_acc = keras.metrics.BinaryAccuracy(name='accuracy')
        a_acc.reset_state()
        a_acc.update_state(a_preds, y_test)
        output['a_acc'] = a_acc.result().numpy()

        a_precision = keras.metrics.Precision(name='precision')
        a_precision.reset_state()
        a_precision.update_state(a_preds, y_test)
        output['a_precision'] = a_precision.result().numpy()

        a_recall = keras.metrics.Recall(name='recall')
        a_recall.reset_state()
        a_recall.update_state(a_preds, y_test)
        output['a_recall'] = a_recall.result().numpy()

        a_auc = keras.metrics.AUC(name='auc')
        a_auc.reset_state()
        a_auc.update_state(a_preds, y_test)
        output['a_auc'] = a_auc.result().numpy()

        output['a_f1'] = f1_score(a_preds, y_test)

        output['best_a'] = best_a
    return output
