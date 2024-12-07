from unimol_tools import MolTrain, MolPredict
import pandas as pd 
import numpy as np
import warnings

#HOMO/LUMO Train
data=np.load('../data/HL_INPUT.npy',allow_pickle=True).item()
data_array=np.arange(0,len(data['atoms']),1)
np.random.seed(55)#455
np.random.shuffle(data_array)
ratio=0.2
N_train=int(ratio*len(data['atoms']))
train_data_HL={'atoms':[data['atoms'][i] for i in data_array[:N_train]],
'coordinates':[data['coordinates'][i] for i in data_array[:N_train]],
'target':[[data['HOMO'][i],data['LUMO'][i]] for i in data_array[:N_train]]}

test_data_HL={'atoms':[data['atoms'][i] for i in data_array[N_train:]],
'coordinates':[data['coordinates'][i] for i in data_array[N_train:]],
'target':[[data['HOMO'][i],data['LUMO'][i]] for i in data_array[N_train:]]}

clf = MolTrain(task='multilabel_regression',
                    data_type='molecule',
                    epochs=40,
                    learning_rate=0.0001,
                    batch_size=16,
                    metrics='mae',
                    split='random',
                    save_path='../module/UnimolHLsave_INPUT',
                    kfold=3,
                    remove_hs=True,
                  )


clf.fit(train_data_HL)

#scan Train
data2=pd.read_csv('../data/INPUT_4.csv')
train_data0 = data.sample(frac=0.9, random_state=1)
train_data=pd.DataFrame(index=range(train_data0.shape[0]),columns=range(20))
train_data.iloc[:,0]=train_data0['smiles'].values
train_data.iloc[:,1:]=train_data0.iloc[:,8:27]
name=['SMILES','TARGET_1', 'TARGET_2', 'TARGET_3', 'TARGET_4', 'TARGET_5', 'TARGET_5', 'TARGET_6', 'TARGET_7', 'TARGET_8', 'TARGET_9', 'TARGET_10','TARGET_11', 'TARGET_12', 'TARGET_13', 'TARGET_14', 'TARGET_15', 'TARGET_16', 'TARGET_17', 'TARGET_18']
train_data.columns=name
train_data.to_csv("../module/Unimolsave/COS2_train.csv", index=False)
test_data0= data.drop(train_data0.index)
test_data=pd.DataFrame(index=range(test_data0.shape[0]),columns=range(20))
test_data.iloc[:,0]=test_data0['smiles'].values
test_data.iloc[:,1:]=test_data0.iloc[:,8:27]#.values.tolist()
test_data.columns=name
test_data.to_csv("./Unimolsave/COS2_test.csv", index=False)
clf = MolTrain(task='multilabel_regression',
                    data_type='molecule',
                    epochs=40,
                    learning_rate=0.0001,
                    batch_size=16,
                    metrics='mae',
                    split='random',
                    save_path='../module/Unimolsave',
                    kfold=5,
                    remove_hs=True,
                  )


clf.fit("../module/Unimolsave/COS2_train.csv")