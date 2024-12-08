"""
6 models, acceleration (accelerometer) and orientation (gyroscope) 

walking vs not walking: accel only, orientation only, both

normal vs abnormal walking: accel only, orientation only, both

optional if time: duck walking vs limping: 
"""
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import pickle
import os.path as osp
# Helper function to visualize model - Do not modify
def viz_tree(dt_model,features_frames,cnames, path):
    # Fix feature names as list
    feature_names = features_frames.columns.tolist()

    fig, ax = plt.subplots(figsize=(9,4))
    tree.plot_tree(dt_model,
                   feature_names=feature_names,
                   fontsize=7,
                   class_names=cnames,
                   filled=True,
                   ax=ax)

    plt.title('Decision Tree')
    plt.savefig(path)
    
def train_decision_tree(frames, name, max_depth=5, path='./models', test_size=0.15):
    # Extract feature columns
    X = frames[['avg', 'max', 'med', 'min', 'q25', 'q75', 'std']]

    # Extract target column
    y = frames['activity']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Create model
    dt_model = DecisionTreeClassifier(criterion='entropy',max_depth=max_depth).fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)

    # Evaluate on test set
    acc = dt_model.score(X_test, y_test)
    dt_cm = confusion_matrix(y_test, dt_pred, labels=dt_model.classes_)
    print(classification_report(y_test, dt_pred))
    #print("Accuracy on test set:", acc)
    print("Accuracy on train set:", dt_model.score(X,y))
    
    with open(osp.join(path, f'{name}.pkl'), 'wb') as f:
        pickle.dump(dt_model, f)
    return dt_model,dt_cm,acc

def evaluate(dt_model, filtered_collected_data):
    X_test = filtered_collected_data[['avg', 'max', 'med', 'min', 'q25', 'q75', 'std']]

    # Extract target column
    y_test = filtered_collected_data['activity']

    dt_pred = dt_model.predict(X_test)
    # Evaluate on test set
    acc = dt_model.score(X_test, y_test)
    # dt_cm = confusion_matrix(y_test, dt_pred, labels=dt_model.classes_)
    print(classification_report(y_test, dt_pred))
    print("Accuracy on test set:", acc)