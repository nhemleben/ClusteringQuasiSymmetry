

#Plots the clusters after they are labeled and saved

import matlab
import matlab.engine
import numpy as np
from scipy.io import netcdf
import sys, os



filenames = [
    '/home/nicholas/Downloads/quasisymmetryLandscape20190224/quasisymmetry_out.20190105-01-021_nfp1_stellSym.nc',
    '/home/nicholas/Downloads/quasisymmetryLandscape20190224/quasisymmetry_out.20190105-01-022_nfp2_stellSym.nc',
    '/home/nicholas/Downloads/quasisymmetryLandscape20190224/quasisymmetry_out.20190105-01-023_nfp3_stellSym.nc',
    '/home/nicholas/Downloads/quasisymmetryLandscape20190224/quasisymmetry_out.20190105-01-024_nfp4_stellSym.nc',
    '/home/nicholas/Downloads/quasisymmetryLandscape20190224/quasisymmetry_out.20190105-01-025_nfp5_stellSym.nc',
    '/home/nicholas/Downloads/quasisymmetryLandscape20190224/quasisymmetry_out.20190105-01-026_nfp6_stellSym.nc',
    '/home/nicholas/Downloads/quasisymmetryLandscape20190224/quasisymmetry_out.20190105-01-027_nfp7_stellSym.nc',
    '/home/nicholas/Downloads/quasisymmetryLandscape20190224/quasisymmetry_out.20190105-01-028_nfp8_stellSym.nc',
    '/home/nicholas/Downloads/quasisymmetryLandscape20190224/quasisymmetry_out.20190105-01-029_nfp9_stellSym.nc',
    '/home/nicholas/Downloads/quasisymmetryLandscape20190224/quasisymmetry_out.20190105-01-030_nfp10_stellSym.nc'
]


multi_R0c1=np.array([])
multi_R0c2=np.array([])
multi_R0c3=np.array([])
multi_Z0s1=np.array([])
multi_Z0s2=np.array([])
multi_Z0s3=np.array([])
multi_eta_bar= np.array([])
multi_rms_curvature = np.array([])

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
    eta_bar= f.variables['scan_eta_bar'][()]
    rms_curvature= f.variables['rms_curvatures'][()]
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


    #going ahead to just look at positive ones
    mask = iotas >= 0.2
    iotas = iotas[mask]

    eta_bar =eta_bar[mask]
    rms_curvature =rms_curvature[mask]

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
    nfps = nfp*np.ones(len(iotas))

    R0c1= R0c[0]
    R0c2=R0c[1]
    R0c3=R0c[2]
    Z0s1=Z0s[0]
    Z0s2=Z0s[1]
    Z0s3=Z0s[2]

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

    multi_eta_bar= np.append(multi_eta_bar,eta_bar)
    multi_rms_curvature= np.append(multi_rms_curvature,rms_curvature)
    multi_R0c1= np.append(multi_R0c1,R0c1)
    multi_R0c2= np.append(multi_R0c2,R0c2)
    multi_R0c3= np.append(multi_R0c3,R0c3)
    multi_Z0s1= np.append(multi_Z0s1,Z0s1)
    multi_Z0s2= np.append(multi_Z0s2,Z0s2)
    multi_Z0s3= np.append(multi_Z0s3,Z0s3)

    multi_nfps = np.append(multi_nfps,nfps)
    multi_dominant_nfps = np.append(multi_dominant_nfps,dominant_nfps)
    multi_helicities = np.append(multi_helicities,helicities)
    multi_max_curvatures = np.append(multi_max_curvatures,max_curvatures)
    multi_max_elongations = np.append(multi_max_elongations,max_elongations)
    multi_standard_deviations_of_R = np.append(multi_standard_deviations_of_R,standard_deviations_of_R)
    multi_standard_deviations_of_Z = np.append(multi_standard_deviations_of_Z,standard_deviations_of_Z)




VarDict= {'eta bar': multi_eta_bar / (np.amax(multi_eta_bar )) ,
          'rms curvatures':  multi_rms_curvature/ (np.amax(multi_rms_curvature )),
          'iotas': multi_iotas/ (np.amax(multi_iotas )),
          'nfps':  multi_nfps / (np.amax( multi_nfps)),
          'dominant nfps':  multi_dominant_nfps/ (np.amax( multi_dominant_nfps)) ,
          'helicities':  np.abs(multi_helicities) / (np.amax(np.abs(multi_helicities ) )), #helicities are all negative
          'max curvatures':  multi_max_curvatures / (np.amax(multi_max_curvatures  )),
          'max elongations': multi_max_elongations / (np.amax(multi_max_elongations  )),
          'std of R':  multi_standard_deviations_of_R/ (np.amax(multi_standard_deviations_of_R )),
          'std of Z':  multi_standard_deviations_of_Z/ (np.amax(multi_standard_deviations_of_Z )),
                    'R0c1':  multi_R0c1 / (np.amax(multi_R0c1 )),
          'R0c2':  multi_R0c2 / (np.amax(multi_R0c2 )),
                    'R0c3':  multi_R0c3 / (np.amax(multi_R0c3 )),
                    'Z0s1':  multi_Z0s1 / (np.amax(multi_Z0s1)),
          'Z0s2':  multi_Z0s2 / (np.amax(multi_Z0s2)),
                    'Z0s3':  multi_Z0s3 / (np.amax(multi_Z0s3)),
          }


InputDict= {
    'eta bar': multi_eta_bar/ (np.amax(multi_eta_bar )),
    'iotas': multi_iotas/ (np.amax( multi_iotas)),
           'R0c1':  multi_R0c1/ (np.amax( multi_R0c1)),
    'R0c2':  multi_R0c2/ (np.amax( multi_R0c2)),
           'R0c3':  multi_R0c3/ (np.amax( multi_R0c3)),
            'Z0s1':  multi_Z0s1/ (np.amax(multi_Z0s1 )),
    'Z0s2':  multi_Z0s2/ (np.amax(multi_Z0s2 )),
           'Z0s3':  multi_Z0s3/ (np.amax(multi_Z0s3 )),
    'nfps':  multi_nfps/ (np.amax(multi_nfps ))
}

#VarDict=InputDict

print "starting matlab"
eng = matlab.engine.start_matlab()

A=[]
for var in VarDict.keys():
    B=np.ndarray.tolist(VarDict[var])
    A.append(matlab.double(B))

T= VarDict.keys()
print "Calling matlab"
eng.ReadAllIn(A, T , len(multi_iotas), nargout=0)


#Since not displaying anything right now
#eng.quit()

#this is just so process does not kill itself before I get a chance to see the plot
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16,7))
plt.show()

