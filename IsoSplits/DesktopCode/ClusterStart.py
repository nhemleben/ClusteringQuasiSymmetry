




#Also want to try out some more plotting tools since matplotlib is o slow


#want to import matlab then have it run iso-split on a batch of dummmy data
#then on a file
#then on all the files
#   since sorting by nfp can do file by file in loop to find clusters

import matlab
import matlab.engine
import numpy as np
from scipy.io import netcdf
import sys, os



#eng = matlab.engine.start_matlab("-desktop")


#B= 1
#c= eng.SimpleJack(B)
#print c

#B= [1 , 2]
#c= eng.SimpleJack(B)
#print c



#Load in a file worth of data

filenames = [
'/home/nicholas/Downloads/quasisymmetryLandscape20190224/quasisymmetry_out.20190105-01-021_nfp1_stellSym.nc', \
'/home/nicholas/Downloads/quasisymmetryLandscape20190224/quasisymmetry_out.20190105-01-022_nfp2_stellSym.nc', \
'/home/nicholas/Downloads/quasisymmetryLandscape20190224/quasisymmetry_out.20190105-01-023_nfp3_stellSym.nc', \
'/home/nicholas/Downloads/quasisymmetryLandscape20190224/quasisymmetry_out.20190105-01-024_nfp4_stellSym.nc', \
'/home/nicholas/Downloads/quasisymmetryLandscape20190224/quasisymmetry_out.20190105-01-025_nfp5_stellSym.nc', \
'/home/nicholas/Downloads/quasisymmetryLandscape20190224/quasisymmetry_out.20190105-01-026_nfp6_stellSym.nc', \
'/home/nicholas/Downloads/quasisymmetryLandscape20190224/quasisymmetry_out.20190105-01-027_nfp7_stellSym.nc', \
'/home/nicholas/Downloads/quasisymmetryLandscape20190224/quasisymmetry_out.20190105-01-028_nfp8_stellSym.nc', \
'/home/nicholas/Downloads/quasisymmetryLandscape20190224/quasisymmetry_out.20190105-01-029_nfp9_stellSym.nc', \
'/home/nicholas/Downloads/quasisymmetryLandscape20190224/quasisymmetry_out.20190105-01-030_nfp10_stellSym.nc']

multi_iotas = np.array([])
multi_nfps = np.array([])
multi_dominant_nfps = np.array([])
multi_helicities = np.array([])
multi_max_curvatures = np.array([])
multi_max_elongations = np.array([])
multi_standard_deviations_of_R = np.array([])
multi_standard_deviations_of_Z = np.array([])
names = []
number_of_experiments = -1

for j in range(len(filenames)):
    filename = filenames[j]
    print "Reading filename "+filename
    f = netcdf.netcdf_file(filename,mode='r',mmap=False)
    iotas = f.variables['iotas'][()]
    nfp = f.variables['nfp'][()]
    max_curvatures = f.variables['max_curvatures'][()]
    max_elongations = f.variables['max_elongations'][()]
    helicities = f.variables['axis_helicities'][()]
    R0c = f.variables['scan_R0c'][()]
    R0s = f.variables['scan_R0s'][()]
    Z0c = f.variables['scan_Z0c'][()]
    Z0s = f.variables['scan_Z0s'][()]
    standard_deviations_of_R = f.variables['standard_deviations_of_R'][()]
    standard_deviations_of_Z = f.variables['standard_deviations_of_Z'][()]
    f.close()




#Sort based on integer valued variables

    basevalNFP = nfp
    basevalHelicities = helicities[1]


#going ahead to just look at positive ones
    mask = iotas >= 0.2
    iotas = iotas[mask]
    max_curvatures = max_curvatures[mask]
    max_elongations = max_elongations[mask]
    helicities = helicities[mask]
    standard_deviations_of_R = standard_deviations_of_R[mask]
    standard_deviations_of_Z = standard_deviations_of_Z[mask]
    R0c = R0c[1:,mask]
    R0s = R0s[1:,mask]
    Z0c = Z0c[1:,mask]
    Z0s = Z0s[1:,mask]
    amplitudes = R0c**2 + R0s**2 + Z0c**2 + Z0s**2
    dominant_nfps = (np.argmax(amplitudes,axis=0)+1)*nfp
    helicities *= nfp
 #   nfps = np.zeros(len(iotas))
    nfps = nfp*np.ones(len(iotas))



#Added an additional filter to force integer values
#to all be the same before sorting
#    mask=[]
#    for i in range(len(iotas)):
#        if basevalHelicities== helicities[i]:
#            mask.append(i)
#    iotas = iotas[mask]
#    max_curvatures = max_curvatures[mask]
#    max_elongations = max_elongations[mask]
#    helicities = helicities[mask]
#    standard_deviations_of_R = standard_deviations_of_R[mask]
#    standard_deviations_of_Z = standard_deviations_of_Z[mask]
#    R0c = R0c[1:,mask]
#    R0s = R0s[1:,mask]
#    Z0c = Z0c[1:,mask]
#    Z0s = Z0s[1:,mask]
#    amplitudes = R0c**2 + R0s**2 + Z0c**2 + Z0s**2
#    dominant_nfps = (np.argmax(amplitudes,axis=0)+1)*nfp
#    helicities *= nfp
#    #   nfps = np.zeros(len(iotas))
#    nfps = nfp*np.ones(len(iotas))



#since doing file at a time, maybe want this outside of loop in future?
#will decide later when this seems less dumb

    if len(iotas)==1:
        # The files that contain a single scan entry are the experiments
        index = filename.find("quasisymmetry_out")
        if index<0:
            print "Error! quasisymmetry_out was not in the filename."
            exit(1)
        names.append(filename[index+18:-3])
    elif number_of_experiments < 0:
        number_of_experiments = len(multi_iotas)

    multi_iotas = np.append(multi_iotas,iotas)
    print "multi_iotas.shape:",multi_iotas.shape
    multi_nfps = np.append(multi_nfps,nfps)
    multi_dominant_nfps = np.append(multi_dominant_nfps,dominant_nfps)
    multi_helicities = np.append(multi_helicities,helicities)
    multi_max_curvatures = np.append(multi_max_curvatures,max_curvatures)
    multi_max_elongations = np.append(multi_max_elongations,max_elongations)
    multi_standard_deviations_of_R = np.append(multi_standard_deviations_of_R,standard_deviations_of_R)
    multi_standard_deviations_of_Z = np.append(multi_standard_deviations_of_Z,standard_deviations_of_Z)

VarDict= {'iotas': multi_iotas,
          'nfps':  multi_nfps ,
          'dominant nfps':  multi_dominant_nfps ,
          'helicities':  multi_helicities ,
          'max curvatures':  multi_max_curvatures ,
          'max elongations':  multi_max_elongations ,
          'std of R':  multi_standard_deviations_of_R ,
          'std of Z':  multi_standard_deviations_of_Z }
#print iotas

#Cluster in Matlab

#eng = matlab.engine.start_matlab()
eng = matlab.engine.start_matlab("-desktop")

#eng.run_test(nargout=0)


#only can check 2d  cluster right now
#just going to do iota and max curvature
#fixed, can now do 3d
A= [1, 2, 3]

#simp_iotas= np.concatenate(multi_iotas)
A[0] = np.ndarray.tolist(multi_iotas)
A[1] = np.ndarray.tolist(multi_max_curvatures)
#A[2] = np.ndarray.tolist(max_elongations)
A[2] = np.ndarray.tolist(multi_nfps)

#eng.helo(A[0],A[1],A[2],nargout=0)


eng.helo(A[0], A[1], A[2], nargout=0)



#c= eng.SimpleJack(np.ndarray.tolist(iotas))
#print c







#eng.quit()


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16,7))
plt.show()
















