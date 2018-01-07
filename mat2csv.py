#!/usr/bin/python

import pandas

data_frame = pandas.read_csv('data/result.txt')

with open('data/result.csv', 'w') as f:
    f.write('price\n')
    for d in data_frame.values:
        f.write('%.0f\n' % d[0])


