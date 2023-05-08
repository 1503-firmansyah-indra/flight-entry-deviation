import os

import numpy as np
import pandas as pd

from utils import get_day_of_week, get_seconds_since_day, train_and_evaluate, get_mean_and_std
from models import Base, Conv, ConvBase, FlattenModel, FlattenBase
import argparse
import tensorflow as tf


def main(sector, oversampling_factor, include_own_pt, normalize_video_occ, buffer, grid_lon, grid_lat, threshold,
         comb, train_vec_model, train_occ_model, occupancy_type, ensemble, learning_rate, epochs, batch_size):
    # load labels
    occ_path = "../processed_data/sector_67Y/occupancy/occs/"
    video_ids_path = "../processed_data/sector_67Y/occupancy/ids.npy"
    own_pt_occ_path = "../processed_data/sector_67Y/occupancy/trajectory.csv"
    labels_path = f'../processed_data/sector_67Y/buffer{buffer}_combined_results.csv'
    if sector == "W":
        occ_path = "../processed_data/sector_w_esmm/occupancy/occs/"
        video_ids_path = "../processed_data/sector_w_esmm/occupancy/ids.npy"
        own_pt_occ_path = "../processed_data/sector_w_esmm/occupancy/trajectory.csv"
        labels_path = f'../processed_data/sector_w_esmm/buffer{buffer}_combined_results.csv'
    labels_and_vec = pd.read_csv(labels_path)
    labels_and_vec_ids = labels_and_vec['id'].values
    nan_indexes = labels_and_vec.loc[labels_and_vec['entry_deviation'].isnull()].index
    labels_and_vec = labels_and_vec.drop(nan_indexes)

    # add time features
    labels_and_vec['day_of_week'] = labels_and_vec['forecasted_entry_time'].apply(get_day_of_week)
    labels_and_vec['seconds_of_day'] = labels_and_vec['forecasted_entry_time'].apply(get_seconds_since_day)

    # Setting threshold to set the label
    labels = labels_and_vec[['id', 'entry_deviation']].copy()
    labels['binary_devi'] = labels['entry_deviation'] > threshold

    print('The proportion of label 1 (deviate): ', (sum(labels['binary_devi']) / len(labels)))

    # further pre-processing
    features = ['fp_entry_lon', 'fp_entry_lat', 'at_pred_lon', 'at_pred_lat', 'day_of_week', 'seconds_of_day']
    X_vec = labels_and_vec[features].values
    y = labels['binary_devi'].values

    # Load Video
    """
    X_video_occ_raw = np.load(occ_path + "1.0.npy")
    for i in range(2, len(os.listdir(occ_path))):
        next_part = np.load(occ_path + f"{i}.0.npy")
        X_video_occ_raw = np.append(X_video_occ_raw, next_part, axis=0)
    """

    X_video_occ_raw = np.load(occ_path + "1.0.npy")
    for i in range(len(os.listdir(occ_path)) - 1):
        try:
            next_part = np.load(occ_path + f"{i + 2}.0.npy")
        except FileNotFoundError:
            continue
        X_video_occ_raw = np.append(X_video_occ_raw, next_part, axis=0)

    video_ids = np.load(video_ids_path)
    filtered_video_array_indexes = []
    for this_id in video_ids:
        if this_id in labels_and_vec_ids:
            filtered_video_array_indexes.append(np.where(video_ids == this_id)[0][0])
    X_video_occ_raw = X_video_occ_raw[filtered_video_array_indexes, :, :]
    assert X_video_occ_raw.shape[0] == labels_and_vec_ids.shape[0]

    N, M, D = X_video_occ_raw.shape
    X_video_occ = []
    for m in range(M):
        time_frames = []
        for n in range(N):
            time_frames.append(np.reshape(X_video_occ_raw[n][m], (grid_lon, grid_lat)))
        X_video_occ.append(time_frames)
    X_video_occ = np.array(X_video_occ)
    X_video_occ = np.delete(X_video_occ, nan_indexes, axis=1)

    if normalize_video_occ:
        mean = np.mean(X_video_occ)
        std = np.std(X_video_occ)
        X_video_occ = (X_video_occ - mean) / std

    if include_own_pt:
        own_pt_occ_df = pd.read_csv(own_pt_occ_path)
        occ_cols = []
        for i in range(grid_lon * grid_lat):
            occ_cols.append(str(i))
        own_pt_occs = []
        for fid in labels['id']:
            this_pt_occ_array = own_pt_occ_df[own_pt_occ_df['flight_id'] == fid][occ_cols].values[0]
            own_pt_occs.append(np.reshape(this_pt_occ_array, (grid_lon, grid_lat)))
        # Channel processing in occupancy matrix
        X_comp_occ = np.append(X_video_occ, [own_pt_occs], axis=0)
    else:
        X_comp_occ = X_video_occ
    X_comp_occ = np.moveaxis(X_comp_occ, 0, -1)

    # Combining Occupancy Matrix
    X = np.array(list(zip(X_vec, X_comp_occ)), dtype=object)

    # Train and evaluate
    seed_list = [11, 30, 2022, 9, 49, 1, 2, 3, 37, 8]
    results = []
    for each_seed in seed_list:
        comb_model = None
        vec_model = None
        occ_model = None
        if comb:
            if occupancy_type == "flatten":
                comb_model = FlattenBase()
            else:
                comb_model = ConvBase()
            comb_model.compile(
                loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        if train_vec_model or ensemble:
            vec_model = Base()
            vec_model.compile(
                loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        if train_occ_model or ensemble:
            if occupancy_type == "flatten":
                occ_model = FlattenModel()
            else:
                occ_model = Conv()
            occ_model.compile(
                loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        res = train_and_evaluate(X=X, y=y,
                                 input_seed=each_seed,
                                 comb_model=comb_model,
                                 vec_model=vec_model,
                                 occ_model=occ_model,
                                 oversampling_factor=oversampling_factor,
                                 ENSEMBLE=ensemble,
                                 epochs=epochs,
                                 batch_size=batch_size)
        results.append(res)
        print(each_seed, "done")
    df_results = pd.DataFrame(results)
    df_results = get_mean_and_std(df_results)
    df_results.to_csv(f"../results/results_nn_{occupancy_type}_sector{sector}.csv", index=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-of", "--oversampling-factor", type=int, default=2,
                        help="Oversampling factor for positive class")
    parser.add_argument("-op", "--own-pt", type=bool, default=True, help="if true the own planned trajectory matrix is "
                                                                         "included in the model input")
    parser.add_argument("-n", "--norm-occ", type=bool, default=False, help="if true the occupancies matrices are "
                                                                           "normalized to have by subtracting the mean "
                                                                           "and dividing by standard deviation")
    parser.add_argument("-b", "--buffer", type=int, default=15,
                        help="the time-wise distance between the projected entry "
                             "point and the time of prediction (binary "
                             "classification) If this is changed respective "
                             "training data has to generated")
    parser.add_argument("-glo", "--gridlon", type=int, default=10,
                        help="size of grid used for occupancy matrices relating to longitude")
    parser.add_argument("-gla", "--gridlat", type=int, default=10,
                        help="size of grid used for occupancy matrices relating to latitude")
    parser.add_argument("-t", "--threshold", type=int, default=5, help="threshold for the binary classification in "
                                                                       "kilometers")
    parser.add_argument("-c", "--comb", type=bool, default=True, help="if true a model using a combination of the "
                                                                      "vector features and occupancy maps will be "
                                                                      "trained and evaluated")
    parser.add_argument("-vo", "--vec-only", type=bool, default=True, help="if true a model using only vector features "
                                                                           "will be trained and evaluated")
    parser.add_argument("-oo", "--occ-only", type=bool, default=True, help="if true a model that only uses the "
                                                                           "occupancy maps / 2-Dimensional input will "
                                                                           "be trained and evaluated")
    parser.add_argument("-ot", "--occ-type", type=str, default="flatten", choices=["flatten", "conv"],
                        help="if true all the occupancy maps will be "
                             "immediatly flattened. Therefore no "
                             "convolutions are used")
    parser.add_argument("-a", "--ensemble", type=bool, default=True, help="if true an ensemble of the vec_only and "
                                                                          "occ_only will be trained and evaluated by "
                                                                          "finding an optimal combination factor (a)")
    parser.add_argument("-e", "--epochs", type=int, default=30, help="number of epochs for the training")
    parser.add_argument("-bs", "--batch-size", type=int, default=200, help="batch-size for model fitting")
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.001, help="learning rate models")
    parser.add_argument("-s", "--sector", type=str, default="W", choices=["67Y", "W"], help="airsector referring to "
                                                                                            "the data")

    ARGS = parser.parse_args()
    print(ARGS)
    main(
        sector=ARGS.sector,
        oversampling_factor=ARGS.oversampling_factor,
        include_own_pt=ARGS.own_pt,
        normalize_video_occ=ARGS.norm_occ,
        buffer=ARGS.buffer,
        grid_lon=ARGS.gridlon,
        grid_lat=ARGS.gridlat,
        threshold=ARGS.threshold,
        comb=ARGS.comb,
        train_vec_model=ARGS.vec_only,
        train_occ_model=ARGS.occ_only,
        occupancy_type=ARGS.occ_type,
        ensemble=ARGS.ensemble,
        learning_rate=ARGS.learning_rate,
        epochs=ARGS.epochs,
        batch_size=ARGS.batch_size
    )
