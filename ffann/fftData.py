
import matplotlib.pyplot as plot
import numpy as np
from numpy.fft import fft, ifft

import statistics

import sys

print("Run this script like: python3 fftData.py filename a|l")

accelUpperCeiling = 300.0
lidarUpperCeiling = 500.0
upperCeiling = 400.0
sr = 4096
histLen = 2048

freqHist = []
for x in range(histLen):
  freqHist.append( 0.0 )

#filename = input("Enter filename to process: ")

filename = sys.argv[1]

if sys.argv[2] == 'a':
  upperCeiling = accelUpperCeiling

if sys.argv[2]  == 'l':
  upperCeiling = lidarUpperCeiling

with open(filename, 'r') as file: 
  sampleBatch = []
  counter = 0
  means15 = []
  maxs15 = []
  maxIndexs15 = []
  modes15 = []
  stddevs = []

  upperMax = 0.0

  for line in file:
    line = line.strip()
    line = line.split(",")
    data = float( line[0] )
    if data > upperCeiling :
      data = upperCeiling #limit to upper ceiling
    data = data / upperCeiling #normalise
    if counter > 2048:
      sampleBatch.append( 0.0 ) #append zeros for better DFT clarity
    else:
      sampleBatch.append( data ) #append normal data up to 2048 individual values
    counter = counter + 1
    if counter > 4096:
      counter = 0
      X = fft( sampleBatch )
      toWrite = np.abs(X[0:2048])
      #fileToWrite = open("fft"+filename, "w")
      for r in range(10,len(toWrite)): # deliberately omitting the 1st 10 values, which skew the results
        if float(toWrite[r]) > upperMax:
          upperMax = float(toWrite[r])
        freqHist[r] = freqHist[r] + toWrite[r]
        print("Just added "+str(toWrite[r])+" to  index "+str(r)+", which now has "+str(freqHist[r]))
      #  fileToWrite.write( str(toWrite[r])+",\n"  )
      N = len( X )
      n = np.arange(N)
      T = N / sr
      freq = n / T
      
      #print("first two freqs are "+str( np.abs(X[0])  )+" and "+str( np.abs(X[1])  ) )
      #print( "Lenght of frequencies is "+str( len(freq)  )+" and freq value 5 is "+str(freq[5]) )
      #print("Also databatch first three vals are: "+str(sampleBatch[0])+" "+str(sampleBatch[1])+" "+str(sampleBatch[2]) )
      #print("Frequences: mean amplitude is "+str( statistics.mean( np.abs(X[1:]) ) )+", and max freq is "+str( max( np.abs(X[1:]) ) )+" at freq "+str(  (np.abs(X[1:])).argmax() )+", and mode amplitude is "+str( statistics.mode((np.abs(X[1:]))) ) )
      '''over5000 = np.abs(X) # np.extract( np.abs(X[1:500]) > 5000, np.abs(X[1:500]) )
      means15.append(statistics.mean( over5000 ))
      maxs15.append(max( over5000 ))
      maxIndexs15.append(over5000.argmax())
      modes15.append( statistics.mode( (over5000) )  )
      stddevs.append( statistics.stdev( over5000) ) 
      '''
      '''
      plot.figure( figsize = (14,7))
      plot.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")
      plot.xlabel('Frequency in Hz')
      plot.ylabel('FFT Amplitude')
      plot.xlim(0,2048)
      plot.show()
      '''

      sampleBatch.clear()

fileToWrite = open("fft"+filename, "w")
for x in freqHist:
  fileToWrite.write( str(x)+",\n"  )

'''
print("Avg of max amps mean and median is "+str( ( statistics.median( maxs15 )  + statistics.mean( maxs15 )  ) / 2.0  ) )
print("Median of max amp is "+str( statistics.median( maxs15 ) )+", mean of max amp is "+str( statistics.mean(maxs15)  )+", mode of max amp: "+str( statistics.mode(maxs15)  )+", and stdev of maxamps: "+str( statistics.stdev(maxs15)  ) )
print("Median of the max frequency is "+str( statistics.median(maxIndexs15) )+", and mean of it is "+str( statistics.mean( maxIndexs15  )  )+", mode of it is"+str( statistics.mode(maxIndexs15)   )+", stdev of it is "+str( statistics.stdev(maxIndexs15)  ) )
print("Median stddev is "+str( statistics.median(stddevs)  )+", mode stddev is "+str( statistics.median(stddevs)  ) )
'''
print("upper max is "+str(upperMax) )
