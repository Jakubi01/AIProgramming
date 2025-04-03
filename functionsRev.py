import numpy as np
import matplotlib.pyplot as plt
import pickle

# 상수 정의
IMAGE_SIZE = 28  # 이미지 크기 (28x28)
NUM_CLASSES = 10  # 클래스 수
NUM_TRAIN_SAMPLES = 300  # trainSet에서 사용할 샘플 수
NUM_TEST_SAMPLES = 100  # testSet에서 사용할 샘플 수

# 데이터 로딩 함수
def init_data():
    with open('train.bin', 'rb') as f1:
        train = pickle.load(f1)

    with open('test.bin', 'rb') as f2:
        test = pickle.load(f2)
    
    return train, test

# 데이터 전처리 함수: 각 데이터셋에서 일부 데이터를 선택하여 준비
def data_ready1(train, test, k = 300):
    trainSet = []
    testSet = []

    for i in range(10):
        if len(train[i]) >= k:
            trainSet.append(train[i][:k])
        else:
            trainSet.append(train[i])  # k보다 적으면 그대로 추가
        if len(test[i]) >= 100:
            testSet.append(test[i][:100])
        else:
            testSet.append(test[i])  # 100보다 적으면 그대로 추가

    return trainSet, testSet

def data_ready2(train, test, k = 300):
    trainSetf = np.zeros((k * 10, 28 * 28))
    testSetf = np.zeros((100 * 10, 28 * 28))

    for i in range(len(train)):
        for j in range(k):
            trainSetf[i * k + j, :] = train[i][j].flatten()

    for i in range(len(test)):
        for j in range(100):
            testSetf[i * 100 + j, :] = test[i][j].flatten()
    
    return trainSetf, testSetf

def knn(trainSet, testSet, k):
    # trainSet과 testSet을 numpy 배열로 변환하여 shape 속성으로 크기 확인
    trainSet = np.array(trainSet)
    testSet = np.array(testSet)

    trS1, trS2 = trainSet.shape
    teS1, teS2 = testSet.shape
    trS3 = int(trS1 / 10)  # 각 클래스의 샘플 수
    teS3 = int(teS1 / 10)  # 각 클래스의 테스트 샘플 수

    label = np.tile(np.arange(0, 10), (teS3, 1))  # 0부터 9까지의 클래스 레이블
    result = np.zeros((teS1, 10))  # 결과 배열 크기를 (teS1, 10)으로 변경

    # 테스트 샘플을 하나씩 처리
    for i in range(teS1):
        # 각 테스트 샘플과 모든 트레인 샘플 간의 거리 계산
        imsi = np.sum((trainSet - testSet[i, :]) ** 2, axis=1)
        
        # 가장 가까운 k개의 인덱스를 찾음
        no = np.argsort(imsi)[:k]
        
        # k개의 인덱스에 대한 클래스 빈도 계산
        hist, bins = np.histogram(no // trS3, bins=np.arange(-0.5, 10.5, 1))
        
        # 가장 많이 나온 클래스를 결과 배열에 저장
        result[i, np.argmax(hist)] = 1  # i번째 테스트 샘플에 대한 예측 클래스 번호 저장
    
    return result

# 템플릿 생성 함수: 각 클래스에 대한 평균 이미지를 계산하여 템플릿을 만듦
def createTmpl(trainSet):
    tmpl = np.zeros((IMAGE_SIZE, IMAGE_SIZE * NUM_CLASSES))  # 템플릿 배열 크기 초기화

    for i in range(NUM_CLASSES):
        # 각 클래스에 대한 평균 이미지 계산
        imsi = np.array(trainSet[i])
        tmpl[:, i * IMAGE_SIZE:(i + 1) * IMAGE_SIZE] = np.mean(imsi, axis=0)

    return tmpl

def tmplMatch(tmpl, testSet):
    result = np.zeros((NUM_TEST_SAMPLES, NUM_CLASSES))
    
    for i in range(len(testSet)):
        for j in range(len(testSet[0])):
            imsiTest = np.tile(testSet[i][j], (1,10))
            error = np.abs(tmpl-imsiTest)
            errorSum = [error[:,0:28].sum(), error[:,28:56].sum(), error[:,56:84].sum(), error[:,84:112].sum(), error [:,112:140].sum(),
        error[:, 140:168].sum(), error [:,168:196].sum(), error[:, 196:224].sum(), error [:,224:252].sum(), error[:,252:280].sum()] 
            
            result[j,i] = np.argmin(errorSum)
            
    return result

def calcMeasure(result):
    s1, s2 = result.shape
    label = np.tile(np.arange(0, s2), (s1, 1))
    
    TP = np.zeros(10)  # TP 배열을 미리 초기화
    TN = np.zeros(10)  # TN 배열을 미리 초기화
    FN = np.zeros(10)  # FN 배열을 미리 초기화
    FP = np.zeros(10)  # FP 배열을 미리 초기화

    for i in range(10):
        TP[i] = ((result == label) & (label == i)).sum()
        TN[i] = ((result != i) & (label != i)).sum()
        FN[i] = ((result != label) & (label == i)).sum()
        FP[i] = ((result == i) & (label != i)).sum()

    acc = (TP + TN) / (TP + TN + FP + FN)
    pre = TP / (TP + FP)
    rec = TP / (TP + FN)
    f1 = 2 * pre * rec / (pre + rec)

    return acc, pre, rec, f1

def feat1(trainSet, testSet):
    trS1 = len(trainSet); trS2 = len(trainSet[0])
    teS1 = len(testSet); teS2 = len(testSet[0])

    trainSetf = np.zeros((trS1 * trS2, 5))
    testSetf = np.zeros((teS1 * teS2, 5))

    for i in range(trS1):
        for j in range(trS2):
            imsi = trainSet[i][j]
            imsi = np.where(imsi != 0)
            imsi2 = np.mean(imsi, 1)
            imsi3 = np.cov(imsi)
            trainSetf[i * trS2 + j, :] = np.array([imsi2[0], imsi2[1], imsi3[0, 0], imsi3[0, 1], imsi3[1, 1]])

    for i in range(teS1):
        for j in range(teS2):
            imsi = testSet[i][j]
            imsi = np.where(imsi != 0)
            imsi2 = np.mean(imsi, 1)
            imsi3 = np.cov(imsi)
            testSetf[i * teS2 + j, :] = np.array([imsi2[0], imsi2[1], imsi3[0, 0], imsi3[0, 1], imsi3[1, 1]])

    return trainSetf, testSetf

def feat2(trainSet, testSet, dX):
    size = trainSet[0][0].shape[0]
    s = size - dX + 1
    trS1 = len(trainSet)
    trS2 = len(trainSet[0])
    teS1 = len(testSet)
    teS2 = len(testSet[0])

    trainImsi = np.zeros((trS1 * trS2, s, s))
    testImsi = np.zeros((teS1 * teS2, s, s))
    trainSetf = np.zeros((trS1 * trS2, s * s))
    testSetf = np.zeros((teS1 * teS2, s * s))

    for i in range(trS1):
        for j in range(trS2):
            imsi = trainSet[i][j]
            for ii in range(s):
                for jj in range(s):
                    if ii + dX <= imsi.shape[0] and jj + dX <= imsi.shape[1]:
                        trainImsi[i * trS2 + j, ii, jj] = imsi[ii: dX + ii, jj: dX + jj].sum()
                trainSetf[i * trS2 + j, :] = trainImsi[i * trS2 + j, ::].flatten()
    
    for i in range(teS1):
        for j in range(teS2):
            imsi = testSet[i][j]
            for ii in range(s):
                for jj in range(s):
                    if ii + dX <= imsi.shape[0] and jj + dX <= imsi.shape[1]:
                        testImsi[i * teS2 + j, ii, jj] = imsi[ii: dX + ii, jj: dX + jj].sum()
                testSetf[i * teS2 + j, :] = testImsi[i * teS2 + j, ::].flatten()

    return trainSetf, testSetf

def data_ready1(train, test, k = 300):
    trainSet = []
    testSet = []

    for i in range(10):
        trainSet.append(train[i][0 : k])
        testSet.append(test[i][0 : 100])

    return trainSet, testSet