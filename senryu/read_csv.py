import csv
import numpy as np

print('start')

X = np.empty((0,30), np.int)
with open('file/csv/senryudb_labeled.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        tmp = []
        for c in list(row[0]):
            tmp.append(np.int.from_bytes(c.encode('utf-8'),'big'))
        x = np.array(tmp)
        print(x)
#        x.resize(30,refcheck=False)
#        print(x)
#        print(list(row[0])
#        X = np.append(X,x,axis=0)

print(X)
print('end')