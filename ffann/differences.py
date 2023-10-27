
# this program will generate a running average of a data series
# ... and output to a file named the same as input with ra_ in front

import sys

upperMax = 0.0

fileToWrite = open("diff_"+sys.argv[1],"w")

buffer = [] # will keep last 10 vals
differenced = []

with open(sys.argv[1],"r") as datafile:

  for line in datafile:
    line = line.strip()
    line = line.split(",")
    data = float( line[0] )
    buffer.append(data)


for x in range(1,len(buffer)):
    diff = buffer[x-1] - buffer[x] # difference between consecutive values
    diff = diff * diff # squrae the difference to make absolute and also accentuate large differences
    differenced.append(diff)
    fileToWrite.write(str(diff)+",\n")
    if diff > upperMax:
      upperMax = diff



print("Largest difference is "+str(upperMax))
