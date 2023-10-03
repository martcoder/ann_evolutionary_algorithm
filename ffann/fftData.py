
import matplotlib.pyplot as plot
import numpy as np
from numpy.fft import fft, ifft

import statistics

import sys

sr = 2000

#filename = input("Enter filename to process: ")

filename = sys.argv[1]

with open(filename, 'r') as file: 
  sampleBatch = []
  counter = 0
  means15 = []
  maxs15 = []
  maxIndexs15 = []
  modes15 = []

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
      
      #print("first two freqs are "+str( np.abs(X[0])  )+" and "+str( np.abs(X[1])  ) )
      #print( "Lenght of frequencies is "+str( len(freq)  )+" and freq value 5 is "+str(freq[5]) )
      #print("Also databatch first three vals are: "+str(sampleBatch[0])+" "+str(sampleBatch[1])+" "+str(sampleBatch[2]) )
      #print("Frequences: mean amplitude is "+str( statistics.mean( np.abs(X[1:]) ) )+", and max freq is "+str( max( np.abs(X[1:]) ) )+" at freq "+str(  (np.abs(X[1:])).argmax() )+", and mode amplitude is "+str( statistics.mode((np.abs(X[1:]))) ) )
      means15.append(statistics.mean( np.abs(X[1:]) ))
      maxs15.append(max( np.abs(X[1:]) ))
      maxIndexs15.append((np.abs(X[1:])).argmax())
      modes15.append( statistics.mode( (np.abs(X[1:])) )  )
      '''
      plot.figure( figsize = (14,7))
      plot.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")
      plot.xlabel('Frequency in Hz')
      plot.ylabel('FFT Amplitude')
      plot.xlim(0,1000)
      #plot.show()
      '''

      sampleBatch.clear()

      

print("Median of the max frequency is "+str( statistics.median(maxIndexs15) )+", and mean of it is "+str( statistics.mean( maxIndexs15  )  ) )
