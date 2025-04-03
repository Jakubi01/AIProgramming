from sklearn.neighbors import KNeighborsClassifier
import functionsRev as fs
import numpy as np
import time

t1 = time.time()
train, test = fs.init_data()


trainSet, testSet = fs.data_ready2(train, test)
label = np.tile(np.arange(0, 10), (300, 1))

knn = KNeighborsClassifier(n_neighbors = 10, weights = "distance", metric = "euclidean")
knn.fit(trainSet, label.T.flatten())
result = knn.predict(testSet)
result = result.reshape(10, 100).T

acc, pre, rec, f1 = fs.calcMeasure(result)
t2 = time.time()

print('클래스 별 성능\n', '[Acc]', '\n', acc,'\n', '[Pre]', '\n', pre, '\n', '[Rec]', '\n', rec, '\n', '[F1]', '\n', f1, '\n')
print('The time required = ', t2 - t1)
print('Acc = ', acc.mean())
print('F1 = ', f1.mean())
