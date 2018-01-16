# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 22:27:49 2017

@author: HAWLET PACKARD
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import math
from numpy import linalg as LA
from scipy.misc import imread, imsave, imresize
from scipy.spatial.distance import pdist, squareform
from skimage import io

DATA_DIR = "YALE/centered/";
TRN_ENUM = ["centerlight","glasses","happy","leftlight","noglasses","normal"];
TEST_ENUM= ["rightlight","sad","sleepy","surprised","wink"];
           
DIR_TOT = 14;
IMG_TOT = 6 * DIR_TOT;

'''
Face Recoginition Using EigenFaces with Manhallobis Distance and 1 Nearest Neighbour.

Progress :
    Read and store image to be next processed for computing eigenvectors + value
    Reduce each image 
    Compute EigenValue and Transpose of EigenVector
    Compute Normalized Image
    Testing Dataset using 1 Nearest Neighbour with Manhallobis Distance
'''

class eigens(object):
    def __init__(self,a,b,c):
        self.eigVal = a;
        self.eigVec = b;
        self.avg = c;

class EigenFaces(object):
    def __init__(self):
        cnt = 1;
        imgCnt = 0;
        
        self.imgList = [];
        
        while(cnt <= DIR_TOT):
            for idx,enm in enumerate(TRN_ENUM):
                print("READING IMAGE NO %d" % (imgCnt+1));
                
                path = "%ssubject%02d.%s.pgm" % (DATA_DIR,cnt,enm);
                img = io.imread(path);
                self.plotImage(img);
                
                if(imgCnt == 0):
                    self.imgShape = img.shape;
                    self.flatList = np.zeros((self.imgShape[0]*self.imgShape[1],IMG_TOT));
                    
                self.imgList.append((str(cnt),img));
                self.flatList[:,imgCnt] = img.flatten();
                imgCnt = imgCnt + 1;
                
            cnt = cnt + 1;
        
        print("DONE READING TRAINING IMAGE");
        self.computeEigen();
        self.testData();
    
    def testData(self):
        cnt = 1;
        imgCnt = 0;
        cntCorrect = 0;
        print("TESTING IMAGE DATA");
        while(cnt <= DIR_TOT):
            
            for idx,enm in enumerate(TEST_ENUM):
                print("TESTING IMAGE NO %d" % (imgCnt+1));
                path = "%ssubject%02d.%s.pgm" % (DATA_DIR,cnt,enm);
                img = io.imread(path);
                
                bef = img;
                img = self.normalizedImg(img.flatten());
                self.showProjectedImage(img);
                
                imgFeature = self.getFeature(img);
                
                mini = -1.0;
                ans = -1;
                idxAns = -1;
                
                for idx,cmp in enumerate(self.imgList):
                    tmp = cmp[1];
                    tmp = self.getFeature(self.normalizedImg(tmp.flatten()));
                    
                    if(idx == 0):
                        mini = self.getDistance(imgFeature,tmp);
                        ans = cmp[0];
                        idxAns = idx;
                    else: 
                        tmp = self.getDistance(imgFeature,tmp);
                        if(mini > tmp):
                            mini = tmp;
                            ans = cmp[0];
                            idxAns = idx;
                
                self.plotAnswerImage(bef, self.imgList[idxAns][1]);
                
                if(ans == str(cnt)):
                    cntCorrect+=1;
                    print("CORRECT");
                else : print("WRONG");
                    
                imgCnt += 1;    
            cnt += 1;
        print("DONE TESTING IMAGE DATA");
        print("Result = %d/%d Correct" % (cntCorrect,imgCnt));
    
    def computeEigen(self):
        print("COMPUTING EIGENVALUES AND EIGENVECTORS");
        avg = self.flatList.mean(axis=1);
        self.plotImage(avg.reshape(self.imgShape));
        for i in range(self.flatList.shape[1]):
            self.flatList[:,i] -= avg;
        
        eigVal,eigVec = np.linalg.eig(np.dot(self.flatList.T,self.flatList));
        eigVec = np.dot(self.flatList,eigVec);
        eigVec = eigVec / np.linalg.norm(eigVec,axis=0);
        
        self.eigen = eigens(eigVal,eigVec,avg);
        self.k = 20;
        print("DONE COMPUTING");
        #EigenValue and Vector are sorted already.        
    
    def plotAnswerImage(self, originalImage, answerImage):
        fig, showImage = plt.subplots(1,2);
        
        showImage[0].set_title("Test Image");
        showImage[0].imshow(originalImage);
        
        showImage[1].set_title("Answer");
        showImage[1].imshow(answerImage);
        plt.show();

    def plotImage(self, img):
        plt.subplot(1,2,1);
        plt.imshow(img);
        plt.show();
    
    def getFeature(self,flattenImg):
        return np.dot(self.eigen.eigVec.T,flattenImg);
    
    def showProjectedImage(self,img):
        a = self.getFeature(img);
        print(type(a[0]));
        imgNew = np.dot(self.eigen.eigVec[:,0:self.k],a[0:self.k]);
        imgNew += self.eigen.avg;
        imgNew = imgNew.reshape(self.imgShape);
        
        img += self.eigen.avg;
        img = img.reshape(self.imgShape);
        
        fig, showImage = plt.subplots(1,2);
        
        showImage[0].set_title("Original Images");
        showImage[0].imshow(img);
        
        showImage[1].set_title("Projected Images");
        showImage[1].imshow(imgNew);
        plt.show();
        
    
    def getDistance(self,a,b):
        dist = 0;
        
        i = 0;
        while(i < self.k):
            dist += (1/self.eigen.eigVal[i])*((a[i]-b[i])**2);
            i += 1;
        
        return dist;
    
    def normalizedImg(self,img):
        return img - self.eigen.avg;
    
EigenFaces();


'''
#RENAME DATA TEST

trg = "assets/";

cnt = 21;


while(True):
    curTrg = trg+str(1);
    a = input("Folder: ");
    if a == "exit" : break;
    takeFrom = "faces94/male/"+str(a)+"/";
    curCnt = 1;
    
    while(curCnt <= 20):
        imageTakeFrom = takeFrom+str(a)+"."+str(curCnt)+".jpg";
        #print(imageTakeFrom);
        img = imread(imageTakeFrom);
        
        saveTrg = trg+str(cnt)+"/"+str(curCnt)+".jpg";
        curCnt = curCnt + 1;
        if (not os.path.exists(trg+str(cnt))):
            os.makedirs(trg+str(cnt));
        
        imsave(str(saveTrg), img);
    
    cnt = cnt+1;


'''