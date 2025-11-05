#### By choutilin1 250630 250825
answer_path  = "/work/choutilin1/out_vars33/vars33_H/out04.nc"
#answer_path  = "/work/choutilin1/data_4d/inf/2017.h5"
predict_path = "/work/choutilin1/out_vars33/vars33_H/out04_perturbed.nc"
var_ids = [2]
#var_ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]  # a list of channel (variable) ids you want to plot
n_frames = 179
lsm_path = None
#lsm_path = "/home/choutilin1/makani/datasets/source/invariant/land_sea_mask.npy"


#### Don't change anything below this ####

minmax = { 0:(None,None),
           1:(None,None),
           2:(96000,106000),
           3:(220,310),
           4:(None,None),
           5:(None,None),
           6:(None,None),
           7:(None,None),
           8:(None,None),
           9:(None,None),
          10:(None,None),
          11:(None,None),
          12:(None,None),
          13:(None,None),
          14:(None,None),
          15:(None,None),
          16:(None,None),
          17:(None,None),
          18:(None,None),
          19:(None,None),
          20:(250,310),
          21:(-2.,1.6),
          22:(None,None),
          23:(None,None),
          24:(None,None),
          25:(None,None),
          26:(None,None),
          27:(None,None),
          28:(None,None),
          29:(None,None),
          30:(None,None),
          31:(None,None),
          32:(None,None),
        }

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
             21:"SSH",
             22:"SSU",
             23:"SSV",
             24:"MLD",
             25:"D15",
             26:"D20",
             27:"SNSH",
             28:"SNLH",
             29:"SNSW",
             30:"SNLW",
             31:"gradx-MSLP",
             32:"grady-MSLP",
            }

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
import netCDF4


if lsm_path is None:
    lsm = np.ones((721,1440))
else:
    lsm = np.load(lsm_path)


#assert( answer_path[-3:]==".h5" )
#answer_data = h5py.File(answer_path)['fields']
assert( answer_path[-3:]==".nc" )
answer_data  = netCDF4.Dataset( answer_path)
assert(predict_path[-3:]==".nc" )
predict_data = netCDF4.Dataset(predict_path)



# for var in var_ids, make a GIF
for var in var_ids:
    print( "making "+var_names[var]+" ..." )
    fig,axes = plt.subplots(nrows=2, dpi=200)
    ax1,ax2 = axes
    #im1 = ax1.imshow(answer_data[1,var,:,:]*lsm, animated=True, vmin=minmax[var][0],vmax=minmax[var][1]) #h5
    im1 = ax1.imshow( answer_data["var"+str(var)][0,:,:]*lsm, animated=True, vmin=minmax[var][0],vmax=minmax[var][1]) #nc
    im2 = ax2.imshow(predict_data["var"+str(var)][0,:,:]*lsm, animated=True, vmin=minmax[var][0],vmax=minmax[var][1])
    ax1.tick_params(labelleft=False, labelbottom=False)
    ax2.tick_params(labelleft=False)
    tmp = ax1.set_ylabel("original")
    tmp = ax2.set_ylabel("perturbed")
    fig.subplots_adjust(hspace=0)
    def animate_func(i):
        #im1.set_data(answer_data[i+1,var,:,:]*lsm) #h5
        im1.set_data( answer_data["var"+str(var)][i,:,:]*lsm) #nc
        ax1.set_title(var_names[var]+f"  time_step{i}")
        im2.set_data(predict_data["var"+str(var)][i,:,:]*lsm)
        return [im1,im2]
    # cbar_ax = fig.add_axes((0.85,0.15,0.05,0.7))
    tmp = fig.colorbar(im1, ax=axes)
    anim = animation.FuncAnimation( fig,
                                    animate_func,
                                    frames=n_frames,
                                    interval=250,
                                    blit=True,
                                    repeat=True )
    writer = animation.PillowWriter( fps=4,
                                     metadata=dict(artist="choutilin1"),
                                     bitrate=1800 )
    anim.save("compare_var"+str(var)+".gif", writer=writer)





