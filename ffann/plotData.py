import matplotlib.pyplot as plot


x = []
rx = []
for z in range(1,101):
  x.append(z)
  rx.append(z)
y = []
ry = []

with open("lineardata.data", 'r') as file:
  for line in file:
    line = line.strip()
    line = line.split(",")
    data = float( line[0] )
    y.append(data)

with open("oneTimeOutput.data", 'r') as fileR:
  for lineR in fileR:
    lineR = lineR.strip()
    lineR = lineR.split(",")
    data = float( lineR[0] )
    ry.append(data)


plot.figure( figsize = (14,7))
plot.plot(x,y, 'b')
plot.plot(rx,ry, 'r')
plot.xlabel('x values')
plot.ylabel('y values')
#plot.xlim(0,499)
plot.legend(['orig values','ann solved values'])
plot.show()

