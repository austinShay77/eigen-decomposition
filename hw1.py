import os
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import cv2

class FileIO:
    def __init__(self):
        pass

    def generateData(self, path):
        X = np.empty(shape=[0, 1600])
        count = 0
        for file in os.listdir(path):
            if not file.endswith(".txt"):
                img_path = os.path.join(path, file)
                img = Image.open(img_path)
                subsampled = img.resize((40, 40))
                img_array = np.array(subsampled)
                flat = img_array.flatten()
                X = np.append(X, [flat], axis=0) 
                count += 1
        return X

class Arithmetic:
    def __init__(self):
        pass

    def unStandardize(self, d, orig):
        return np.around(d * np.std(orig, axis=0, ddof=1) + np.mean(orig, axis=0))

    def standardize(self, d):
        return (d - np.mean(d, axis=0)) / (np.std(d, axis=0, ddof=1))
    
    def covariance(self, d):
        return np.cov(d, rowvar=False)  # features are col
    
    def eigan(self, d):
        return np.linalg.eigh(d)
    
    def projectionMatrix(self, eiganValues, eiganVectors, k):
        p = []
        working = np.array(eiganValues)
        for _ in range(0, k):
            mostRelevant = np.argmax(working)
            working[mostRelevant] = -math.inf
            p.append(eiganVectors[:, mostRelevant])
        return np.array(p).transpose()

def hw1(path):
    fileio = FileIO()
    arithmetic = Arithmetic()
    data = fileio.generateData(path)
    standard = arithmetic.standardize(data)
    covMatrix = arithmetic.covariance(standard)
    eiganValues, eiganVectors = arithmetic.eigan(covMatrix)

    # part 2
    w = arithmetic.projectionMatrix(eiganValues, eiganVectors, 2)
    pca = np.dot(standard, w)
    plt.plot(pca[:, 0], pca[:, 1], 'mo')
    plt.savefig("part2.png")

    # part 3
    video = cv2.VideoWriter("part3.avi", cv2.VideoWriter_fourcc(*"XVID"), 15, (40, 40), False)
    for k in range(1, len(eiganValues)+1):
        w = arithmetic.projectionMatrix(eiganValues, eiganVectors, k)
        pca = np.dot(standard, w)
        standardReconstruction = np.dot(pca, w.transpose())
        reconstruction = arithmetic.unStandardize(standardReconstruction, data)
        reshape = reconstruction[102].reshape((40, 40)).astype(np.uint8) # grabs subject02.centerlight
        img_filename = './temp/' + str(k) + '.png'
        Image.fromarray(reshape).save(img_filename, 'PNG')
        image = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)
        cv2.putText(image, str(k), (1, 38), cv2.FONT_HERSHEY_PLAIN, .5, (0, 0, 0), 1)
        video.write(image)
        os.remove(img_filename)
    video.release()

if __name__ == "__main__":
    if not os.path.exists('./temp'):
        os.makedirs("./temp")
    hw1("./yalefaces")
    os.rmdir("./temp")