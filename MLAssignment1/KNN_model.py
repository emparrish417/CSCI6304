import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image
import PIL
import numpy as np
import math
import time

# def eval_model():
#
#     mod_label
#
#     recall = tp / (fn + tp)
#     precision = tp / (fp + tp)
#     true_neg_rate = fp / (tn + fp)
#
#     precision_i = i / sum_a
#     recall_i = i / sum_p
#
#     return tp, tn, fp, fn

# def plot_data(data):
#
#     data.plot(x='Age', y=['Glucose','BloodPressure','BMI'], kind='scatter')
#     plt.show()


def plot_image(data):

    sample_list = [10, 3, 60, 45, 12, 22, 300, 150, 57]

    for i in sample_list:
        raw_list = mnist_data.iloc[i].to_list()[1:]
        image_list = list()
        for item in raw_list:
            new_entry = (item, item, item)
            image_list.append(new_entry)

        img_size = 28

        img = Image.new("RGB", (img_size,img_size))

        pixels = img.load()

        for x in range(img_size):
            for y in range(img_size):
                pixels[x,y] = image_list[img_size*x+y]

        img.save(os.path.join(out_data_path, "sample_img_" + str(i) +  ".png"))

def minmax_scaling(data):
    new_df = pd.DataFrame()

    for column in data:
        col_data = np.array(data[column])
        new_data = (col_data - min(col_data)) / (max(col_data) - min(col_data))

        new_df[column] = new_data

    return new_df

def standard_scaling(data):
    new_df = pd.DataFrame()

    for column in data:
        col_data = np.array(data[column])
        new_data = (col_data - np.mean(col_data) / np.std(col_data))

        new_df[column] = new_data

    return new_df


def split_train_test(data, frac):

    train_df = data.sample(frac=frac)
    test_df = pd.concat([train_df, data]).drop_duplicates(keep=False)

    return train_df, test_df


def distance_calc(a,b):
    distance = 0

    for i in range(0, len(a) - 1):

        distance = float(distance) + (float((a[i]) - float(b[i])) ** 2)

    final_distance = math.sqrt(distance)

    return final_distance

def find_NN(train, test, nn):

    d = dict()
    nb = list()

    for i in train:
        train_instance = i[1].to_list()
        dist = distance_calc(test, train_instance)
        d[dist] = train_instance

    dist_list = list(d.keys())
    dist_list.sort()

    for j in range(0, nn):
        distance = dist_list[j]
        instance = d[distance]
        nb.append(instance)

    return nb


def knn_mnist(data):
    train, test = split_train_test(data, 0.8)

    outfile = open(r'outputs/minst.csv', 'w')
    outfile.write()

    test_labels = test['label'].to_list()
    test = test.drop('label', 1)
    predictions = list()
    train_labels = train['label']
    train = train.drop('label', 1)
    train['label'] = train_labels

    count = 0
    corr_count = 0
    st_time = time.process_time()
    train_list = train.iterrows()
    print = (type(train_list))

    for i in test.iterrows():
        test_instance = i[1].to_list()
        neighbors = find_NN(train_list, test_instance, 3)
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)
        if count % 100 == 0:
            print(str(count) + ' test rows processed')
            print('Process Time: ' + str((time.process_time() - st_time)))

        #print(str(prediction) + ' ' + str(test_labels[count]))

        if math.isnan(prediction):
            prediction = 0

        if int(prediction) == int(test_labels[count]):
            corr_count += 1

        count += 1

    print('Accuracy: ' + str(corr_count / count))


def knn_pima(data):
    scaled_data = standard_scaling(data)
    train, test = split_train_test(data, 0.9)

    test_labels = test['Outcome'].to_list()
    test = standard_scaling(test.drop('Outcome', 1))
    predictions = list()
    train_labels = train['Outcome']
    train = standard_scaling(train.drop('Outcome', 1))
    train['Outcome'] = train_labels

    count = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in test.iterrows():
        test_instance = i[1].to_list()
        neighbors = find_NN(train, test_instance, 2)
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)
        if math.isnan(prediction):
            prediction = 0

        if int(test_labels[count]) == 1:
            if int(prediction) == 1:
                tp += 1
            elif int(prediction) == 0:
                fn += 1
        elif int(test_labels[count]) == 0:
            if int(prediction) == 1:
                fp += 1
            elif int(prediction) == 0:
                tn += 1

        count += 1


    print('Accuracy: ' + str((tp + tn) / count))
    print('True Positive: ' + str(tp))
    print('False Positive: ' + str(fp))
    print('True Negative: ' + str(tn))
    print('False Negative: ' + str(fn))


in_data_path = r'input_data/KNN'
out_data_path = r'outputs'

start = time.process_time()

mnist_data = pd.read_csv(os.path.join(in_data_path,'train.csv'))
load_mnist = time.process_time()
print(load_mnist - start)
pima_data = pd.read_csv(os.path.join(in_data_path,'diabetes.csv'))
load_pima = time.process_time()
print(load_pima - load_mnist)


knn_mnist(mnist_data)
mnist_time = time.process_time()
print(mnist_time - load_mnist)

#plot_data(pima_data)

knn_pima(pima_data)
pima_time = time.process_time()
print(pima_time)

#standard_scaling(pima_data)
#plot_image(mnist_data)