
import math

x = 0.0
y = 0.0
c = 1.4
m = 0.7

with open("lineardata.data","w") as myfile:
  for x in range(100):
    y = (m * x) + c
    #y = math.sin(math.radians(x))
    x = x + 1.0 
    myfile.write(str(y)+",\n")
