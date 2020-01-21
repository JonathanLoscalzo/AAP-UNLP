import numpy as np
import matplotlib.pyplot as plt 

def prob2labels(y_prob):
    n,c=y_prob.shape
    labels=np.zeros(n,dtype=int)
    labels = y_prob.argmax(axis=1)
    return labels

def labels2onehot(labels):
    n,=labels.shape
    classes=int(labels.max()+1)
    onehot_matrix=np.zeros((n,classes),dtype=int)
    
    #     for i, l in zip(range(n), labels): 
    #         onehot_matrix[i,l]=1
    onehot_matrix[range(n), labels] = 1
    
    return onehot_matrix

def onehot2labels(onehot_matrix):
    n,c=onehot_matrix.shape
    labels = np.zeros(n,dtype=int)
    
    labels = onehot_test.argmax(axis=1)
    
    return labels

def plot_curve(history, ind):
# summarize history for loss
    plt.plot(history.history[ind])
    if ("val_"+ind in history.history.keys()):
          plt.plot(history.history["val_"+ind])
    # plt.plot(history.history['val_loss'])
    plt.title(f'model {ind}')
    plt.ylabel(ind)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def calculate_class_weight(y):
    weights = {}
    for i in np.unique(y):
        weights[i] = ((y == i).sum()) / y.size
    return weights

class Rescaler:
    def __init__(self):
        self.mean = 0
        self.std = 0
        self.fitted = False
        
    def fit(self, X):
        
        self.mean = X.mean()
        self.std = X.std()
        self.fitted = True
    
    def transform(self, X):
        if (not self.fitted):
            raise Exception('Debes ejecutar fit primero')
        return (X - self.mean) / self.std