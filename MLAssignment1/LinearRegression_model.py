import numpy as np
import pandas as pd
import os



in_data_path = r'input_data/LinearRegression'
out_data_path = r'outputs'

student_data = pd.read_csv(os.path.join(in_data_path,'student-por.csv'))

def encode_cats(data):
    for i in data.columns.to_list():

        if data[i].dtype == object:
            data[i] = data[i].astype('category')
            data[i] = data[i].cat.codes

    return data

new = encode_cats(student_data)
print(new.corr())

