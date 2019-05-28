######################################## Imports
import numpy as np
########################################

class MLR: # multiple logistic regression
######################################## public methods

#################### constructor for model
    def __init__(self, learning_rate, itter_num):
        self.learning_rate = learning_rate
        self.itter_num=itter_num
####################   

####################
    def Fit(self, features, labels):
        ##### organize labels
        list_labels=self.__Count_Labels(labels)
        labels=self.__Update_Labels(labels, list_labels)

        ##### init weights
        self.w = np.random.normal(0, 1, (np.size(list_labels),features[0].size))
        self.b = np.random.normal(0, 1, (np.size(list_labels),))

        ########## itterations
        itter=0
        cost_arr=np.zeros(self.itter_num)
        while 1:
            ##### calculate parameters of foward propagation
            output = self.__Foward_Propagation_All_Features(features, list_labels)

            ##### calculate cost
            cost_arr[itter]=self.__Calc_Cost(labels, output)

            ##### calculate grad
            w_grad, b_grad = self.__Calc_Grad(features, labels)

            ##### update parameters
            self.w-=w_grad * self.learning_rate
            self.b-=b_grad * self.learning_rate

            itter+=1
            ##### break from loop
            if itter==self.itter_num:
                break

        return cost_arr
####################

####################
    def Predict(self, input):
        z_sig, _=self.__Foward_Propagation(input)
        output=np.argmax(z_sig)
        return output    
####################

####################
    def Evaluate(self, features, labels):
        count_success=0
        num_of_features=np.size(labels)
        output=np.zeros(num_of_features)
        for i in range(num_of_features):
            cur_input=features[i]
            output[i] = self.Predict(cur_input)
            if(output[i]==labels[i]):
                count_success+=1
        success_rate=(count_success/num_of_features)*100
        return success_rate
####################

######################################## private methods

####################
    def __Calc_Grad(self, features, labels):
        ##### size
        features_num, input_size=np.shape(features)
        _,output_size=np.shape(labels)
        ##### init
        w_grad=np.zeros((output_size,input_size))
        b_grad=np.zeros((output_size,))

        for output_size_i in range(output_size):
            for input_size_i in range(input_size + 1 ): # +1 for b
                for features_num_i in range(features_num):
                    ##### cur parameters
                    cur_input=features[features_num_i,:]
                    z_sig, z=self.__Foward_Propagation(cur_input)
                    #
                    cur_z_sig=np.array([z_sig[output_size_i]])
                    cur_z=np.array([z[output_size_i]])
                    #
                    cur_labels=labels[features_num_i]
                    cur_label=cur_labels[output_size_i]
                    

                    ##### mul
                    mul1=self.__Dif_Cost(cur_z_sig, cur_label, features_num)
                    mul2=self.__Dif_Sigmoid(cur_z)
                    all_mul=mul1*mul2

                    ##### last mul
                    if input_size_i!=input_size:
                        last_dif=cur_input[input_size_i]
                        w_grad[output_size_i, input_size_i]+=all_mul*last_dif
                    else: # for b
                        last_dif=1
                        b_grad[output_size_i]+=all_mul*last_dif

        return w_grad, b_grad
####################

####################
    def __Dif_Cost(self, input, label, features_num):
        output=2*(1/features_num)*(input-label)
        return output
####################

####################
    def __Dif_Sigmoid(self, input):
        output=self.__Sigmoid(input)*(1-self.__Sigmoid(input))
        return output
####################

####################
    def __Calc_Cost(self, labels, output):
        features_num,_=np.shape(labels)
        cost=0
        for i in range(0,features_num):
            cur_output=output[i]
            cur_label=labels[i]
            cost+=self.__Calc_Cost_Single_Output(cur_output, cur_label)
        cost/=features_num
        return cost
####################

####################
    def __Calc_Cost_Single_Output(self, output, labels):
        cost=0
        for i in range(np.size(output)):
            cur_output=output[i]
            cur_label=labels[i]
            cost+=self.__MSE(cur_output, cur_label)
        cost/=np.size(output)
        return cost
####################

####################
    def __MSE(self, calc_output, label):
        cost=(calc_output-label)
        cost=np.power(cost,2)
        return cost
####################

####################
    def __Foward_Propagation_All_Features(self, features, list_labels):
        features_num,_=np.shape(features)
        output=np.zeros((features_num, len(list_labels)))
        for i in range(features_num):
            input=features[i,:]
            output[i], _=self.__Foward_Propagation(input)
        return output
####################        

####################
    def __Foward_Propagation(self, input):
        z=np.matmul(self.w,input)+self.b
        z_sig=self.__Sigmoid(z)
        return z_sig, z
####################

####################
    def __Sigmoid(self, x):
        y=np.zeros(np.size(x))
        for i in range (0,np.size(x)):
            y[i]=1/(1+ np.exp(-x[i]) )
        return y
####################

####################
    def __Count_Labels(self, labels):
        list_labels =[]
        list_labels.append(labels[0])
        for i in range(np.size(labels)):
            if( not(labels[i] in list_labels) ):
                list_labels.append(labels[i])
        list_labels.sort()
        return list_labels
####################

####################
    def __Update_Labels(self, labels, list_labels):
        labels_out=np.zeros((np.size(labels),np.size(list_labels)))
        for i in range(np.size(labels)):
            labels_out[i][labels[i]]=1
        return labels_out
####################

####################
    def __Init_List(self, n):
        list=[]
        for i in range(0,n):
            list.append(0)
        return list
####################


        
    