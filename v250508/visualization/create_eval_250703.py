#### By choutilin1 250703
answer_path  = "/work/choutilin1/out_vars22a/targ_0005.nc"
predict_path = "/work/choutilin1/out_vars22a/out_0005.nc"
var_ids = [3,20,21]  # a list of channel (variable) ids you want to compute
do_RMSE = True
do_ACC  = False



#### Don't change anything below this ####

time_means_path = "/work/choutilin1/out_vars22a/time_means_2011-2014.npy"

var_names = { 0:"U10M",
              1:"V10M",
              2:"MSLP",
              3:"T2M",
              4:"SP",
              5:"U1000",
              6:"V1000",
              7:"Z1000",
              8:"T850",
              9:"U850",
             10:"V850",
             11:"Z850",
             12:"R850",
             13:"T500",
             14:"U500",
             15:"V500",
             16:"Z500",
             17:"R500",
             18:"Z50",
             19:"TCWV",
             20:"SST",
             21:"SSH",}

import numpy as np
import matplotlib.pyplot as plt
import h5py
import netCDF4


time_means = np.load(time_means_path)[0]  # (1, 22, 721, 1440)
assert( answer_path[-3:]==".nc" )
assert(predict_path[-3:]==".nc" )
answer_data  = netCDF4.Dataset( answer_path)
predict_data = netCDF4.Dataset(predict_path)



# for var in var_ids, compute RMSE and ACC
for var in var_ids:
    print( "making "+var_names[var]+" ..." )
    answer  = np.array( answer_data["var"+str(var)])
    predict = np.array(predict_data["var"+str(var)])
    #
    if do_RMSE:
        RMSE = (predict-answer)**2  # (20, 721, 1440)
        # average over time
        plt.imshow(np.sqrt(np.mean(RMSE,axis=0)))
        plt.colorbar()
        plt.title(var_names[var]+"  RMSE")
        plt.show()
        # average over space
        plt.plot(np.sqrt(np.mean(RMSE,axis=(1,2))),'o')
        plt.title(var_names[var]+"  RMSE")
        plt.xlim(0,25)
        plt.show()
    #
    if do_ACC:
        A =  answer-time_means[var]
        P = predict-time_means[var]
        # time step 0
        covAA = A[0,:,:]*A[0,:,:]
        covPP = P[0,:,:]*P[0,:,:]
        covAP = A[0,:,:]*P[0,:,:]
        for t in range(1,20):  # time steps 1~19
            covAA += A[1,:,:]*A[1,:,:]
            covPP += P[1,:,:]*P[1,:,:]
            covAP += A[1,:,:]*P[1,:,:]
        ACC = covAP / np.sqrt(covAA*covPP)
        plt.imshow(ACC); plt.colorbar(); plt.title(var_names[var]+"  ACC"); plt.show()
    #




