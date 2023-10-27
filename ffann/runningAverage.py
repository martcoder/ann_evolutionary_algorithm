
# this program will generate a running average of a data series
# ... and output to a file named the same as input with ra_ in front

import sys

upperMax = 0.0

with open(sys.argv[1],"r") as datafile:


  buffer = [] # will keep last 10 vals
  bufferLen = 5
  counter = 0
  fileToWrite = open("ra_"+sys.argv[1],"w")
  for line in datafile:
    line = line.strip()
    line = line.split(",")
    data = float( line[0] )
    buffer.append(data)
    if counter >= bufferLen:
      buffer = buffer[1:] # drop the first aka oldest item
      dataWrite = 0.0
      for x in range(bufferLen):
        dataWrite = dataWrite + buffer[x] # sum last 10 vals
      dataWrite = dataWrite / float(bufferLen) 
      if dataWrite > upperMax:
        upperMax = dataWrite
      fileToWrite.write(str(dataWrite)+",\n")
    counter = counter + 1

print("Largest value is "+str(upperMax))
