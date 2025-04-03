import functionsRev as fs
import time

t1 = time.time()

# 데이터 초기화 및 준비
train, test = fs.init_data()
trainSet, testSet = fs.data_ready1(train, test)

dX = 20
k = 10
trainSetf1, testSetf1 = fs.feat2(trainSet, testSet, dX)

# KNN을 사용하여 결과 계산
result = fs.knn(trainSetf1, testSetf1, k)

# 성능 평가
acc, pre, rec, f1 = fs.calcMeasure(result)

t2 = time.time()

# 결과 출력
print("The time required : ", t2 - t1)
print('k = ', k)
print('dX = ', dX)
print('Acc = ', acc.mean())
print('F1 = ', f1.mean())