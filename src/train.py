from preprocesser import *
from model import *
import argparse
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--length', default=5)
    parser.add_argument('-sr', '--sampling_rate', default=100)
    parser.add_argument('-d', '--max_depth', default=5)
    parser.add_argument('-n', '--name')
    parser.add_argument('-g', '--gyroscope', action='store_true')
    parser.add_argument('-a', '--accelerometer', action='store_true')
    args = parser.parse_args()
    
    
    print(args)
    
    
    if not (args.gyroscope or args.accelerometer):
        args.gyroscope = True
        args.accelerometer = True
    sensors = []
    if args.gyroscope:
        sensors.append('gyroscope')
    if args.accelerometer:
        sensors.append('accelerometer')

    all_feature_dfs = []

    for sensor in sensors:
        activities = [f for f in os.listdir('./data') if not f.startswith('.')]
        print(activities)
        sensor_feature_dfs = []
        for activity in activities:
            combined_dataframe = CSV_Builder(sample_rate=100).add_csv_files_in_directory(
                relative_path=osp.join("./data/", activity),
                keyword= sensor[1:],
                trim=True
            ).finish_build()

            final_df = Preprocessor(combined_dataframe, args.sampling_rate).calc_magnitude("accel_mag", remove_gravity=False).remove_noise("accel_mag", "filtered_accel_mag").to_date_time().finish_build()

            feature_df = FeatureExtractor(final_df, args.sampling_rate, args.length).extract_basic_features(column_name="filtered_accel_mag", sensorname=sensor, activity=activity).finish_build()

            sensor_feature_dfs.append(feature_df)
        sensor_feature_dfs = pd.concat(sensor_feature_dfs)
        all_feature_dfs.append(sensor_feature_dfs)
        
    all_feature_dfs = pd.concat(all_feature_dfs, axis=1)
    print(all_feature_dfs.head(10))
    print(all_feature_dfs.tail(10))
    dt_model, dt_cm, dt_acc = train_decision_tree(feature_df, args.name, max_depth=args.max_depth)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = dt_cm, display_labels=dt_model.classes_)
    cm_display.plot()
    plt.show()