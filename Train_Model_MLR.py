######################################## pwd
import sys
pwd=sys.path[0]
########################################

######################################## Imports
import numpy as np
import pickle
import matplotlib.pyplot as plt
###
import tkinter as tk
from tkinter import filedialog
########################################

######################################## Import from folder
folder_name='Functions'
path_name_new=pwd + '\\' + folder_name
sys.path.append(path_name_new)
##### Import Files
import Main_Func, Classification_Algorithms
from Classification_Algorithms import MLR
########################################

################################################################################
################################################################################
################################################################################

##### get user input
print('Choose train set')
file_name_path = filedialog.askopenfilename()
file_name=Main_Func.Get_Name_From_File_Path(file_name_path)
print('Trained set is:', file_name)
print('-----')

##### load sets
filehandler  = open(file_name_path,'rb')
set = pickle.load(filehandler)
filehandler.close()

##### unpack
features, labels=Main_Func.Unpack_Set(set)

##### flat set
features_flat = features.reshape(features.shape[0], features[0].size)

##### create model
print('Training model \n . \n . \n .')
model=MLR(learning_rate=3, itter_num=100)
cost_arr = model.Fit(features_flat, labels)
print('Model trained')

##### save model
file_name='Model_MLR'
file_name_path=pwd+'\\'+ file_name +'.obj'
filehandler = open(file_name_path,"wb")
pickle.dump(model,filehandler)
filehandler.close()
print('Model saved to file:',file_name)

##### evaluate
success_rate= model.Evaluate(features_flat, labels)
print('Success Rate=', ' %0.2f ' % success_rate ,'%')

##### plot cost
plt.plot(cost_arr)
plt.xlabel('Itter')
plt.ylabel('Cost')
plt.title('Cost VS Itter')
plt.show()

### hold cmd
print('--------------------')
input('Press Enter to exit...')



