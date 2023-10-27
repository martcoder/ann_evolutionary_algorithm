import matplotlib.pyplot as plot
import sys

x = []
rx = []
for z in range(1,181):
  x.append(z)
  rx.append(z)
y = []
ry = []

plotOne = 0

if( len(sys.argv) > 1):
 filename = sys.argv[1]
 plotOne = 1
 print("Plotting just this one file of data")
else:
 filename = "lineardata.data"

with open(filename, 'r') as file:
  for line in file:
    line = line.strip()
    line = line.split(",")
    data = float( line[0] )
    y.append(data)

if plotOne == 1:
  x.clear()
  for z in range(len(y)):
    x.append(z+1)

if plotOne == 0:
 with open("oneTimeOutput.data", 'r') as fileR:
   for lineR in fileR:
    lineR = lineR.strip()
    lineR = lineR.split(",")
    data = float( lineR[0] )
    ry.append(data)


plot.figure( figsize = (14,7))
plot.plot(x,y, 'b')
if plotOne == 0:
 plot.plot(rx,ry, 'r')


plot.xlabel('x values')
plot.ylabel('y values')
#plot.xlim(0,499)
plot.legend(['orig values','ann solved values'])
plot.show()

