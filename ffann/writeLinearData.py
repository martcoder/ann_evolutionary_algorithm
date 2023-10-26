
import math
import numpy as np

x = 0.0
y = 0.0
c = 1.4
m = 0.3
#xvals = np.linspace(0.0,0.07,100)
counter = 0

with open("lineardata.data","w") as myfile:
    
  while counter < 180:
    counter = counter + 1
    #for ind in range(len(xvals)):
    #x = xvals[ind]
    #y = (m * x) + c
    y = math.sin(math.radians(x))
    #y = (0.0073 * x) - (6e-06)
    # I'M GONNA SCALE UP THE VALUES... THEN DESCALE AFTER :) :) :)
    #y = y * 1000.0 # scaling up by 1000 to help the algorithm!!
    x = x + 1.0 
    myfile.write(str(y)+",\n")
