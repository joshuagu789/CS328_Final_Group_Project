from preprocesser import *
from model import *
import argparse
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--length', default=5, help='Resample window length in seconds')
    parser.add_argument('-sr', '--sampling_rate', default=100)
    parser.add_argument('-d', '--max_depth', default=5)
    parser.add_argument('-n', '--name', default='gait detection', help='Training name, model and images stored at models/name.')
    parser.add_argument('-g', '--gyroscope', action='store_true', help='Use -g to train on gyroscope data only.')
    parser.add_argument('-a', '--accelerometer', action='store_true', help='Use -a to train on accelerometer data only.')
    args = parser.parse_args()
    
    path = './data' # To train for walking vs not walking
    # path = './data/walking' # To train for abnormal vs normal walking
    # path = './data/walking/abnormal' # To train for limp vs duck walking
    
    out_dir = f'./models/{args.name}'
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    
    print(f'Training Hyperparams: \n{args}')
    
    if not (args.gyroscope or args.accelerometer):
        args.gyroscope = True
        args.accelerometer = True
        
    sensors = []
    if args.gyroscope:
        sensors.append('gyroscope')
    if args.accelerometer:
        sensors.append('accelerometer')

    all_feature_dfs = []
    
    i = 0
    for sensor in sensors:
        activities = [f for f in os.listdir(path) if not f.startswith('.')]
        sensor_feature_dfs = []
        for activity in activities:
            fns = sorted(glob.glob(f'{path}/{activity}/**/*{sensor}*.csv', recursive=True))
            for fn in fns:
                df = pd.read_csv(fn)
                processed_df = Preprocessor(df, args.sampling_rate).calc_magnitude("accel_mag", remove_gravity=False).remove_noise("accel_mag", "filtered_accel_mag").to_date_time().finish_build()
                feature_df = FeatureExtractor(processed_df, args.sampling_rate, args.length).extract_basic_features(column_name="filtered_accel_mag", sensorname=sensor, activity=activity).finish_build()
                sensor_feature_dfs.append(feature_df)
                
        sensor_feature_dfs = pd.concat(sensor_feature_dfs)
        sensor_feature_dfs.reset_index(inplace=True, drop=True) # use drop = False to check if multiple sensor data matches
        if(i > 0):
            sensor_feature_dfs.drop(['activity'], axis=1, inplace=True)    
        all_feature_dfs.append(sensor_feature_dfs)
        i += 1
        
    all_feature_dfs = pd.concat(all_feature_dfs, axis=1)
    
    features = [feature for feature in list(all_feature_dfs.keys()) if feature != 'activity']
    
    print(f'Features used in training: {features}')

    dt_model, dt_cm, dt_acc = train_decision_tree(all_feature_dfs, args.name, features, max_depth=args.max_depth, path=f'./models/{args.name}/')
    cm_display = ConfusionMatrixDisplay(confusion_matrix = dt_cm, display_labels=dt_model.classes_)
    cm_display.plot()
    plt.show()
    cm_display.figure_.savefig(f'./models/{args.name}/cm.png', dpi=200)
    viz_tree(dt_model, all_feature_dfs, activities, f'./models/{args.name}/dtree.png')