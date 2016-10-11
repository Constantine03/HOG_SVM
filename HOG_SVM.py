import cv2, numpy as np, os,math

class StatModel(object):
    '''parent class - starting point to add abstraction'''    
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)
class SVM(StatModel):
    '''wrapper for OpenCV SimpleVectorMachine algorithm'''
    def __init__(self):
        self.model = cv2.SVM()

    def train(self, samples, responses, params):
        
        
        self.model.train(samples, responses, params = params)

    def predict(self, samples):
        return np.float32( [self.model.predict(s) for s in samples])

def imagePreparation(image):
    h,w = tuple(image.shape[:2])
    croppedImage = image[(h-picHeight)/2:(h+picHeight)/2, (w-picWidth)/2:(w+picWidth)/2]
    return croppedImage
def obtainGrads(img):
    #getting gradients
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
 
    mag, ang = cv2.cartToPolar(gx, gy)

    return mag, ang
def obtainHist(mag,ang):
    #binning
    bins = []
    temp = []
    for i in range(len(ang)):
        for j in range(len(ang[0])):
            if ang[i][j] > np.pi:
                temp.append(int(numOfBins*(ang[i][j]-np.pi)/(np.pi)))
            else:
                temp.append(int(numOfBins*ang[i][j]/(np.pi)))
        bins.append(temp)
        temp = []

    
    bin_cells = [bins[i][0:sizeOfCell] for i in range(sizeOfCell)], [bins[i][0:sizeOfCell] for i in range(sizeOfCell,sizeOfBlock)], [bins[i][sizeOfCell:sizeOfBlock] for i in range(sizeOfCell)], [bins[i][sizeOfCell:sizeOfBlock] for i in range(sizeOfCell,sizeOfBlock)]
    mag_cells = [mag[i][0:sizeOfCell] for i in range(sizeOfCell)], [mag[i][0:sizeOfCell] for i in range(sizeOfCell,sizeOfBlock)], [mag[i][sizeOfCell:sizeOfBlock] for i in range(sizeOfCell)], [mag[i][sizeOfCell:sizeOfBlock] for i in range(sizeOfCell,sizeOfBlock)]

    hists = [np.bincount([x for sub in b for x in sub], [y for subb in m for y in subb], numOfBins) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist
def Norm(hist,norm):
    if norm == 'L2-hys':
        L2Norm=math.sqrt(sum(map(lambda x:x*x,hist))+e*e)
        hisNorm1=map(lambda x:x/L2Norm,hist)
        hisNorm2=[i if i<0.2 else 0.2 for i in hisNorm1]
        L2NormHys=math.sqrt(sum(map(lambda x:x*x,hisNorm2))+e*e)
        return map(lambda x:float(x/L2NormHys),hisNorm2)
    elif norm == 'L2':
        L2Norm=math.sqrt(sum(map(lambda x:x*x,hist))+e*e)
        return map(lambda x:x/L2Norm,hist)
    elif norm == 'L1':
        L1Norm=sum(map(lambda x:abs(x),hist))+e
        return map(lambda x:x/L1Norm,hist)
    elif norm == 'L1-sqrt':
        L1Norm=sum(map(lambda x:abs(x),hist))+e
        return map(lambda x:math.sqrt(x/L1Norm),hist)
    else:
        raise NameError('Norm name error!')
def obtainPictureHoG(img):
    mag, ang = obtainGrads(img)
    descriptor=[]
    histograms=[]
    for row in range(0,len(img)-sizeOfBlock+1,sizeOfBlock/4):
        for col in range(0,len(img[0])-sizeOfBlock+1,sizeOfBlock/4):
            #range of pixels for given cell
            rowRange=range(row,row+sizeOfBlock)
            colRange=range(col,col+sizeOfBlock)
            #select required data from magnitudes and angles
            cellAng=[]
            temp=[]
            for i in rowRange:
                for j in colRange:
                    temp.append(ang[i,j])
                cellAng.append(temp)
                temp=[]
            cellMag=[]
            for i in rowRange:
                for j in colRange:
                    temp.append(mag[i,j])
                cellMag.append(temp)
                temp=[]
            #get HoGs for given cell
            histograms.append(obtainHist(cellMag,cellAng))
            #get histogram normalizations
            descriptor.append(Norm(histograms[-1],'L1-sqrt'))
    return [item for sublist in descriptor for item in sublist]
def trainModelSamples(path):
    #Take pictures for training
    
    trainPositive = os.listdir(path+'pos/')
    trainNegative = os.listdir(path+'neg/')
    labelPositive = [1 for i in range(len(trainPositive))]
    labelNegative = [0 for i in range(len(trainNegative))]
    picTrainPos = [imagePreparation(cv2.imread(path+'pos/'+pic,0)) for pic in trainPositive]
    picTrainNeg = [imagePreparation(cv2.imread(path+'neg/'+pic,0)) for pic in trainNegative]
    trainUnion = trainPositive+trainNegative
    labelUnion = labelPositive+labelNegative
    picTrainUnion = picTrainPos+picTrainNeg
    #Obtain HoG for train sample
    trainHoG = map(obtainPictureHoG,picTrainUnion)
    #Convert data to suitable format
    trainHoG05 = np.array(trainHoG,dtype=np.float32)
    labelUn = np.array(labelUnion,dtype = np.float32)
    return trainHoG05,labelUn
def trainingSVM(trainHoG05,labelUn,params,name,svm):
    #Train model
    svm.train(trainHoG05,labelUn,params)
    #Check model on train sample
    train_val = svm.predict(trainHoG05)
    print 'Correctness on train sample equals '+str(sum([1.0 if train_val[i]==labelUn[i] else 0.0 for i in range(len(labelUn))])/len(labelUn))
    #Save model for further usage
    svm.save(name)
    return zip(train_val,labelUn)
def testModelSamples(path):
    #Load needed model

    testPositive = os.listdir(path+'pos/')
    testNegative = os.listdir(path+'neg/')
    labelPositiveT = [1 for i in range(len(testPositive))]
    labelNegativeT = [0 for i in range(len(testNegative))]
    picTestPos = [imagePreparation(cv2.imread(path+'pos/'+pic,0)) for pic in testPositive]
    picTestNeg = [imagePreparation(cv2.imread(path+'neg/'+pic,0)) for pic in testNegative]
    testUnion = testPositive+testNegative
    labelUnionT = labelPositiveT+labelNegativeT
    picTestUnion = picTestPos+picTestNeg
    #Obtain HoG for train sample
    testHoG = map(obtainPictureHoG,picTestUnion)
    
    #Convert data to suitable format
    testHoG05 = np.array(testHoG,dtype=np.float32)
    labelUnT = np.array(labelUnionT,dtype = np.float32)
    return testHoG05, labelUnT
def testingSVM(testHoG05, labelUnT, name,svm):
    svm.load(name)
    test_val = svm.predict(testHoG05)
    print 'Correctness on test sample equals '+str(sum([1.0 if test_val[i]==labelUnT[i] else 0.0 for i in range(len(labelUnT))])/len(labelUnT))
    return zip(test_val,labelUnT)
def runKernel(trainHoG,trainLabels,testHoG,testLabels,kernel,svm_type,C,gamma,degree,coef0,clas_name):
    print clas_name[:-4]
    params = dict (kernel_type = kernel, svm_type = svm_type, C = C, gamma = gamma, degree = degree, coef0 = coef0)
    svm = SVM()
    trainingSVM(trainHoG, trainLabels,params, clas_name,svm)
    testingSVM(testHoG, testLabels, clas_name,svm)
#Main     
picHeight = 128
picWidth = 64
sizeOfCell = 8
sizeOfBlock = 16
numOfBins = 9
e=0.001
pathToTrain = 'img/Train/'
pathToTest = 'img/Test/'
trainHoG, trainLabels = trainModelSamples(pathToTrain)
testHoG, testLabels = testModelSamples(pathToTest)
#Try different kernels
runKernel(trainHoG,trainLabels, testHoG, testLabels, cv2.SVM_LINEAR, cv2.SVM_C_SVC,0.125,1,1,1,'svm_human_linear_0_125.dat')
runKernel(trainHoG,trainLabels, testHoG, testLabels, cv2.SVM_LINEAR, cv2.SVM_C_SVC,0.0625,1,1,1,'svm_human_linear_0_0625.dat')
runKernel(trainHoG,trainLabels, testHoG, testLabels, cv2.SVM_LINEAR, cv2.SVM_C_SVC,0.03125,1,1,1,'svm_human_linear_0_03125.dat')
runKernel(trainHoG,trainLabels, testHoG, testLabels, cv2.SVM_LINEAR, cv2.SVM_C_SVC,0.015625,1,1,1,'svm_human_linear_0_015625.dat')                                 
runKernel(trainHoG,trainLabels, testHoG, testLabels, cv2.SVM_LINEAR, cv2.SVM_C_SVC,0.0078125,1,1,1,'svm_human_linear_0_0078125.dat')
runKernel(trainHoG,trainLabels, testHoG, testLabels, cv2.SVM_LINEAR, cv2.SVM_C_SVC,0.001,1,1,1,'svm_human_linear_0_001.dat')
runKernel(trainHoG,trainLabels, testHoG, testLabels, cv2.SVM_LINEAR, cv2.SVM_C_SVC,0.0001,1,1,1,'svm_human_linear_0_0001.dat')
runKernel(trainHoG,trainLabels, testHoG, testLabels, cv2.SVM_LINEAR, cv2.SVM_C_SVC,0.0007,1,1,1,'svm_human_linear_0_0007.dat')
runKernel(trainHoG,trainLabels, testHoG, testLabels, cv2.SVM_LINEAR, cv2.SVM_C_SVC,0.0008,1,1,1,'svm_human_linear_0_0008.dat')
runKernel(trainHoG,trainLabels, testHoG, testLabels, cv2.SVM_LINEAR, cv2.SVM_C_SVC,0.0009,1,1,1,'svm_human_linear_0_0009.dat')
runKernel(trainHoG,trainLabels, testHoG, testLabels, cv2.SVM_RBF, cv2.SVM_C_SVC,1,0.005,1,1,'svm_human_gamma_1_0_005.dat')
runKernel(trainHoG,trainLabels, testHoG, testLabels, cv2.SVM_RBF, cv2.SVM_C_SVC,1,0.0005,1,1,'svm_human_gamma_1_0_0005.dat')
runKernel(trainHoG,trainLabels, testHoG, testLabels, cv2.SVM_RBF, cv2.SVM_C_SVC,0.5,2,1,1,'svm_human_gamma_0_5_2.dat')
runKernel(trainHoG,trainLabels, testHoG, testLabels, cv2.SVM_RBF, cv2.SVM_C_SVC,0.5,1,1,1,'svm_human_gamma_0_5_1.dat')
runKernel(trainHoG,trainLabels, testHoG, testLabels, cv2.SVM_RBF, cv2.SVM_C_SVC,0.5,0.05,1,1,'svm_human_gamma_0_5_0_05.dat')
runKernel(trainHoG,trainLabels, testHoG, testLabels, cv2.SVM_RBF, cv2.SVM_C_SVC,0.5,0.005,1,1,'svm_human_gamma_0_5_0_005.dat')
runKernel(trainHoG,trainLabels, testHoG, testLabels, cv2.SVM_RBF, cv2.SVM_C_SVC,0.5,0.0005,1,1,'svm_human_gamma_0_5_0_0005.dat')                            
runKernel(trainHoG,trainLabels, testHoG, testLabels, cv2.SVM_RBF, cv2.SVM_C_SVC,0.5,0.0006,1,1,'svm_human_gamma_1_0_0006.dat')
runKernel(trainHoG,trainLabels, testHoG, testLabels, cv2.SVM_RBF, cv2.SVM_C_SVC,0.5,0.0007,1,1,'svm_human_gamma_1_0_0007.dat')
runKernel(trainHoG,trainLabels, testHoG, testLabels, cv2.SVM_RBF, cv2.SVM_C_SVC,0.5,0.0008,1,1,'svm_human_gamma_1_0_0008.dat')
runKernel(trainHoG,trainLabels, testHoG, testLabels, cv2.SVM_RBF, cv2.SVM_C_SVC,0.5,0.0009,1,1,'svm_human_gamma_1_0_0009.dat')
