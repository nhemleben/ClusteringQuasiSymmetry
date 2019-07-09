

#Plots the clusters after they are labeled and saved

import matlab
import matlab.engine
import numpy as np
from scipy.io import netcdf
import sys, os

print "starting matlab"
eng = matlab.engine.start_matlab()
#eng = matlab.engine.start_matlab("-desktop")

print "Calling matlab"
eng.NoReadIn(nargout=0)

#this is just so process does not kill itself before I get a chance to see the plot
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(1,7))
plt.show()


