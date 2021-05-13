
from flask import Flask, request
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib
from keras.models import Sequential
import os
import tensorflow as tf
import keras
import math

app = Flask(__name__)


classifier1 = None
classifier2 = None
classifier3 = None
classifier4 = None
graph = tf.get_default_graph()
session = tf.Session()

@app.route('/', methods=['GET', 'POST'])
def download_file():
    global graph
    global session
    try:
        if request.method == 'POST':
            f = request.get_json()
            dt = convert_to_pd(f)
            global classifier1
            global classifier2
            global classifier3
            global classifier4
            if classifier1 is not None:
                model1 = classifier1
                model2 = classifier2
                model3 = classifier3
                model4 = classifier4
            else:
                with session.as_default():
                    with graph.as_default():
                        model1 = joblib.load(os.path.dirname(os.path.realpath(__file__)) + '/models/model_1.pkl')
                        model2 = joblib.load(os.path.dirname(os.path.realpath(__file__)) + '/models/model_2.pkl')
                        model3 = joblib.load(os.path.dirname(os.path.realpath(__file__)) + '/models/model_3.pkl')
                        model4 = joblib.load(os.path.dirname(os.path.realpath(__file__)) + '/models/model_4.pkl')
                classifier1 = model1
                classifier2 = model2
                classifier3 = model3
                classifier4 = model4
            #model_path = os.path.abspath(os.getcwd())+'/models/'
            #model1 = joblib.load(os.path.dirname(os.path.realpath(__file__)) + '/models/model_1.pkl')
            #model2 = joblib.load(os.path.dirname(os.path.realpath(__file__)) + '/models/model_2.pkl')
            #model3 = joblib.load(os.path.dirname(os.path.realpath(__file__)) + '/models/model_3.pkl')
            #model4 = joblib.load(os.path.dirname(os.path.realpath(__file__)) + '/models/model_4.pkl')


            pred = {}
            pred['1'] = predict_for_nn(dt, model1)
            pred['2'] = predict_for_nn(dt, model2)
            pred['3'] = predict_for_nn(dt,model3)
            pred['4'] = predict_for_nn(dt,model4)
            print('result is '+str(pred))
            return pred

    except Exception as e:
        print("Exception while running __fit_model()" + str(e))
        raise e




def convert_to_pd(jsonStr):
    columns = ['score_overall', 'nose_score', 'nose_x', 'nose_y', 'leftEye_score', 'leftEye_x', 'leftEye_y',
               'rightEye_score', 'rightEye_x', 'rightEye_y', 'leftEar_score', 'leftEar_x', 'leftEar_y',
               'rightEar_score', 'rightEar_x', 'rightEar_y', 'leftShoulder_score', 'leftShoulder_x', 'leftShoulder_y',
               'rightShoulder_score', 'rightShoulder_x', 'rightShoulder_y', 'leftElbow_score', 'leftElbow_x',
               'leftElbow_y', 'rightElbow_score', 'rightElbow_x', 'rightElbow_y', 'leftWrist_score', 'leftWrist_x',
               'leftWrist_y', 'rightWrist_score', 'rightWrist_x', 'rightWrist_y', 'leftHip_score', 'leftHip_x',
               'leftHip_y', 'rightHip_score', 'rightHip_x', 'rightHip_y', 'leftKnee_score', 'leftKnee_x', 'leftKnee_y',
               'rightKnee_score', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x', 'leftAnkle_y',
               'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y']
    data = jsonStr
    # data = json.loads(open(path_to_video + 'key_points.json', 'r').read())
    csv_data = np.zeros((len(data), len(columns)))
    for i in range(csv_data.shape[0]):
        one = []
        one.append(data[i]['score'])
        for obj in data[i]['keypoints']:
            one.append(obj['score'])
            one.append(obj['position']['x'])
            one.append(obj['position']['y'])
        csv_data[i] = np.array(one)
    return pd.DataFrame(csv_data, columns=columns)


def predict_for_nn(file_name, model):
    global graph
    global session
    cnt = len(file_name.index) // 30
    df = pd.DataFrame()
    i = 0
    while i < len(file_name.index) - cnt + 1:
        temp = file_name.iloc[i]
        newrow = {}
        newrow['re_dist'] = getdist(temp['rightElbow_x'], temp['rightElbow_y'], temp['nose_x'], temp['nose_y'])
        newrow['le_dist'] = getdist(temp['leftElbow_x'], temp['leftElbow_y'], temp['nose_x'], temp['nose_y'])
        newrow['rw_dist'] = getdist(temp['rightWrist_x'], temp['rightWrist_y'], temp['nose_x'], temp['nose_y'])
        newrow['lw_dist'] = getdist(temp['leftWrist_x'], temp['leftWrist_y'], temp['nose_x'], temp['nose_y'])
        newrow['rs_dist'] = getdist(temp['rightShoulder_x'], temp['rightShoulder_y'], temp['nose_x'], temp['nose_y'])
        newrow['ls_dist'] = getdist(temp['leftShoulder_x'], temp['leftShoulder_y'], temp['nose_x'], temp['nose_y'])
        newdf = pd.DataFrame(newrow, index=[0])
        df = pd.concat([df, newdf])

        # df.append(f[i:i+cnt,:].mean())
        i += cnt

    df.drop(df.columns[0], axis=1)
    X = df.head(30).values

    # min_max_scaler = preprocessing.MinMaxScaler()
    # X = min_max_scaler.fit_transform(file_values)

    features_set = []
    features_set.append(X)
    features_set = np.array(features_set)
    features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 6))
    with session.as_default():
        with graph.as_default():
            predictions = model.predict(features_set)

    labelencoder = joblib.load(os.path.dirname(os.path.realpath(__file__)) + '/models/lencoder.pkl')
    onehotencoder = joblib.load(os.path.dirname(os.path.realpath(__file__)) + '/models/ohencoder.pkl')

    rew1 = onehotencoder.inverse_transform([predictions[0]])
    resp = labelencoder.inverse_transform([int(rew1[0][0])])[0]

    return resp


def getdist(x1, y1, x2, y2):
    value = math.sqrt((float(x2) - float(x1)) ** 2 + (float(y2) - float(y1)) ** 2)
    return value

def predict_for_input(file_name, model):
    file_values = file_name.values

    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(file_values)
    #
    predictions = model.predict(x_scaled)
    # writer = csv.writer(infile)

    f = collections.defaultdict(int)
    for result in predictions:
        if result == 1:
            f['buy'] += 1
            # print('buy is',f['buy'] )
        elif result == 2:
            f['communicate'] += 1
            # print('com is',f['com'] )
        elif result == 3:
            f['fun'] += 1
        elif result == 4:
            f['hope'] += 1
        elif result == 5:
            f['mother'] += 1
        else:
            f['really'] += 1

    print(f)

    return max(f.items(), key=operator.itemgetter(1))[0]


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
