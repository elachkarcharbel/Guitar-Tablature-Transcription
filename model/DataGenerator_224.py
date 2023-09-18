import numpy as np
from tensorflow import keras
import cv2

class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, list_IDs, data_path="../data/spec_repr/", batch_size=128, shuffle=True, label_dim = (6,21), spec_repr="c", con_win_size=9):
        
        self.list_IDs = list_IDs
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.label_dim = label_dim
        self.spec_repr = spec_repr
        self.con_win_size = con_win_size
        self.halfwin = con_win_size // 2
        
        if self.spec_repr == "c":
            self.X_dim = (self.batch_size, 224, 224, 1)            
        self.y_dim = (self.batch_size, self.label_dim[0], self.label_dim[1])
        
        self.on_epoch_end()
        
    def __len__(self):
        # number of batches per epoch
        return int(np.floor(float(len(self.list_IDs)) / self.batch_size))
    
    def __getitem__(self, index):
        # generate indices of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        # generate data
        X, y = self.__data_generation(list_IDs_temp)
        
        return X, y
    
    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, list_IDs_temp):
        #Generates data containing batch_size samples
        # X : (n_samples, *dim, n_channels)
        
        # Initialization
        X = np.empty(self.X_dim)
        y = np.empty(self.y_dim)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            # determine filename
            data_dir = self.data_path + self.spec_repr + "/"
            filename = "_".join(ID.split("_")[:-1]) + ".npz"
            frame_idx = int(ID.split("_")[-1])
            
            # load a context window centered around the frame index
            loaded = np.load(data_dir + filename)
            full_x = np.pad(loaded["repr"], [(self.halfwin,self.halfwin), (0,0)], mode='constant')
            sample_x = full_x[frame_idx : frame_idx + self.con_win_size]

            ## stretching images to create squared shape of 192x192
            ## stretching images to create squared shape of 224x224
            stretched_sample = np.repeat(sample_x, 22, axis=0)
            stretched_sample = np.resize(stretched_sample, (192,192))
            stretched_sample = cv2.resize(stretched_sample, dsize=(224, 224))
            X[i,] = np.expand_dims(np.swapaxes(stretched_sample, 0 ,1), -1)

            # Store label
            y[i,] = loaded["labels"][frame_idx]

        return X, y
