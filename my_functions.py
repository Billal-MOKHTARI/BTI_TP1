#Ce fichier sera alloué aux fonctions de la partie II du TP 01
import matplotlib.pyplot as plt
import cv2
import numpy as np

def OpenImage(src):
    I = plt.imread(src).astype('float32')
    L = I.shape[0]
    C = I.shape[1]
    return I, L, C

def Divide(I):
    C1 = I[:,:, 0]
    C2 = I[:,:,1]
    C3 = I[:,:,2]

    return C1, C2, C3


def HSV(src):
    image = cv2.imread(src, 3)
    image_HSV = cv2.cvtColor (image, cv2.COLOR_RGB2HSV)

    return image_HSV

def CountPix(I):
    N = I.shape[0]*I.shape[1]

    return N
    

def FactPix(I, a, b):

    return np.add(a*I, b)

def Func_a(I):
    return np.log(I), np.exp(I), np.square(I), np.sqrt(I)

def Func_m(I):
    N = CountPix(I)*3

    mean = np.sum(np.sum(I))/N
    std = np.sqrt(np.sum(np.sum(np.add(I, -mean)**2))/N)
    return mean, std

# Pas encore
def Normalize(I, new_min, new_max):
    L = new_max-new_min
    max = np.max(np.max(I))
    min = np.min(np.min(I))

    I_norm = (L/max-min)*(I-min)

    return I_norm

def Inverse(I):
    Inv = np.max(I)-I

    return Inv

# Pas encore
def CalcHist(I):
    I_gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    
    unique_values = np.unique(I_gray, return_counts=True)
    H = unique_values[1]
    b = unique_values[0]
    return H, b

def Threshold(I, threshold):
    image = np.copy(I)
    image[I >= threshold] = 255.0
    image[I < threshold] = 0.0

    return image
    

def Func_j(src):
    I = OpenImage(src)[0]
    H, b = CalcHist(I)

    plt.figure(figsize=(15, 8))
    plt.title("Original image")
    plt.subplot(1, 2, 1)
    plt.imshow(I)
    plt.subplot(1, 2, 2)
    plt.bar(b, H)

    I_inv = Inverse(I)
    H_inv, b_inv = CalcHist(I_inv)

    plt.figure(figsize=(15, 8))
    plt.title("Processed image")
    plt.subplot(1, 2, 1)
    plt.imshow(I_inv)
    plt.subplot(1, 2, 2)
    plt.bar(b_inv, H_inv)

def Func_t(src, min=10/256, max=50/256):
    I = OpenImage(src)[0]
    H, b = CalcHist(I)

    plt.figure(figsize=(15, 8))
    plt.title("Original image")
    plt.subplot(1, 2, 1)
    plt.imshow(I)
    plt.subplot(1, 2, 2)
    plt.bar(b, H)

    I_norm = Normalize(I, min, max)
    H_norm, b_norm = CalcHist(I_norm)

    plt.figure(figsize=(15, 8))
    plt.title("Processed image")
    plt.subplot(1, 2, 1)
    plt.imshow(I_norm)
    plt.subplot(1, 2, 2)
    plt.bar(b_norm, H_norm)

def Func_f(src, threshold=128.0/256.0):
    I = OpenImage(src)[0]
    H, b = CalcHist(I)

    plt.figure(figsize=(15, 8))
    plt.title("Original image")
    plt.subplot(1, 2, 1)
    plt.imshow(I)
    plt.subplot(1, 2, 2)
    plt.bar(b, H)

    I_thresh = Threshold(I, threshold)
    H_thresh, b_thresh = CalcHist(I_thresh)

    plt.figure(figsize=(15, 8))
    plt.title("Processed image")
    plt.subplot(1, 2, 1)
    plt.imshow(I_thresh)
    plt.subplot(1, 2, 2)
    plt.bar(b_thresh, H_thresh)
