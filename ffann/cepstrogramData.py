
import matplotlib.pyplot as plot
import numpy as np
from numpy.fft import fft, ifft

import statistics

import sys

sr = 1000

#filename = input("Enter filename to process: ")

filename = sys.argv[1]

upperMax = 0.0

upperCeiling = 50000.0

with open(filename, 'r') as file: 
  sampleBatch = []
  counter = 0
  means15 = []
  maxs15 = []
  maxIndexs15 = []
  modes15 = []
  firstVal15 = []
  stddevs = []

  for line in file:
    line = line.strip()
    line = line.split(",")
    data = float( line[0] )
    sampleBatch.append( data )
    counter = counter + 1
    if counter > 1999:
      counter = 0
      X = fft( sampleBatch )
      N = len( X )
      n = np.arange(N)
      T = N / sr
      freq = n / T
      #over5000 = np.extract( np.abs(X[1:500]) > 5000.0 , np.abs(X[1:500]) )
      squared = np.power(X[1:1000],2) #square
      logged = np.log(squared) # take natural log
      C = np.power( fft( logged ), 2 ) #fft again, then finally square again

      toWrite = np.abs(C)
      fileToWrite = open("cep"+filename, "a")
      floatToWrite = 0.0
      for r in range(len(toWrite)):
        if float(toWrite[r]) > upperMax:
          upperMax = float(toWrite[r])
        if float(toWrite[r]) > upperCeiling:
          floatToWrite = upperCeiling
        else:
          floatToWrite = toWrite[r]
        fileToWrite.write( str(floatToWrite)+",\n"  )

      #print("first two freqs are "+str( np.abs(X[0])  )+" and "+str( np.abs(X[1])  ) )
      #print( "Lenght of frequencies is "+str( len(freq)  )+" and freq value 5 is "+str(freq[5]) )
      #print("Also databatch first three vals are: "+str(sampleBatch[0])+" "+str(sampleBatch[1])+" "+str(sampleBatch[2]) )
      #print("Frequences: mean amplitude is "+str( statistics.mean( np.abs(X[1:]) ) )+", and max freq is "+str( max( np.abs(X[1:]) ) )+" at freq "+str(  (np.abs(X[1:])).argmax() )+", and mode amplitude is "+str( statistics.mode((np.abs(X[1:]))) ) )
      '''means15.append(statistics.mean( np.abs(C) ))
      maxs15.append(max( np.abs(C) ))
      maxIndexs15.append((np.abs(C)).argmax())
      modes15.append( statistics.mode( (np.abs(C)) )  )
      firstVal15.append( np.abs(C[0])  )
      stddevs.append( statistics.stdev(np.abs(C) ) )
      '''
      '''
      print("size of C is "+str(len(np.abs(C)))+" and first 3 vals are "+str( np.abs(C)[0])+" and "+str(np.abs(C[1]) )+" and "+str( np.abs(C[2])  ) )
      plot.figure( figsize = (14,7))
      plot.stem(freq[1:1000], np.abs(C), 'b', markerfmt=" ", basefmt="-b")
      plot.xlabel('Time')
      plot.ylabel('QFrequency')
      plot.xlim(0,500)
      plot.show()
      '''

      sampleBatch.clear()

      
'''
print("Median of the max Cep values is "+str( statistics.median(maxs15) )+", and mean of it is "+str( statistics.mean( maxs15  )  )+", and mode of maxes is "+str( statistics.mode(maxs15)  ) )
#print("Median of the mode Cep values is "+str( statistics.median(modes15) )+", and mean of it is "+str( statistics.mean( modes15  )  ) )
print("Stddev of firstval is "+str( statistics.stdev(firstVal15)  )+", Mean of firstval is "+str( statistics.mean(firstVal15) )+", median is "+str( statistics.median(firstVal15)  )+", mode is "+str( statistics.mode(firstVal15)  ) )
#print("Stdev avg is "+str( statistics.mean(stddevs)  )+", and median is "+str( statistics.median(stddevs)  ) )
'''
print("Printed cepstrogram values to file with upperCeiling of "+str(upperCeiling)+". Max value found was "+str(upperMax))
