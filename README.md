# ann_evolutionary_algorithm

===PROJECT TITLE=================

Artificial Neural Network Training System using an Evolutionary Algorithm


===LICENSE=======================

The software in this repository may be distributed under the Lesser GPL License, which can be found in this 
same repository. 


===BACKGROUND====================
- Primer on Evolutionary Algorithms (from a different author): https://www.section.io/engineering-education/introduction-to-evolutionary-algorithms/ 
- This project was created intially as part of a state-estimation project which use accelerometer and LiDAR data to try to infer tyre pressure on 
  a vehicle. The data files included in this repo are therefore accel and lidar data which was generated through real-world journeys from a sensor
  rig which was attached to a vehicle... however any data can be used in their place.
  The data filenames are hardcoded in the C code (so this would need manually altering if you want to use other data filenames), but their format is
  expected to be the data value followed by a comma for each line, and the C code reads up to the first comma for each line of the data file before
  converting each into a datapoint. 


===PROJECT OVERVIEW==============
This project trains an artificial neural network by making use of an 
evolutionary algorithm, which creates a population or randomly-initialised 
ANNs, then processes data from a datafile through each ANN, provides each 
a score, then uses the found score to create the next generation through 
evolutionary techniques such as elitism, breeding through tournaments, 
and mutation with probability. 

The algorithm itself is implemented in C. The artifical neural networks are feed-forward ANNs, therefore the code 
so far is included in the FFANN folder. 

The algorithm has been implemented to allow two different modes of functionality:
Mode0- linear regression which uses ANNs with only one output node
Mode1- tri-classification of data which uses softmax and argmax on each ANN's three output nodes. 

===CODE NOTES====================
C does not allow lambda declarations so all possible structs are pre-declared. This means that population sizes are currently limited 
to 199 although this can be increased by simply including more declarations in declarations_aux.c.
 
Some python scripts are present as they were useful in automatically generating the hundreds of struct 
declarations (the approach taken was to declare the maximum amount of structs needed, then use up to that maximum when executing). 
Python is also used for generating data to test with, and also for plotting graphs of results using matplotlib. 

The main for-loop which loops over each member of a population is implemented to use OpenMP parallel-processing across 4 cores, 
with this number of cores hard-coded, so number of cores needs adjusting manually. 

Memory allocated using malloc is freed at the end of main in algorithm.c  

===COMPILING AND RUNNING=========

    C CODE FILES:-  algorithm.c, algorithm_aux.c, and declarations_aux.c
      
    Python CODE FILES:- plotData.py  writeLinearData.py and more.... PLEASE HAVE MATPLOTLIB library installed. 

    Bash CODE FILES:- compile.sh  runAlgo.sh
            
    ENVIRONMENT:- The project was built on linux for ease of access to a C compiler. 

    COMPILING:--- Included in the repository is a compile.sh shell script, which can be ran using: 
                  source compile.sh

                  However to manually compile you can use the following on linux: 
                  gcc -fopenmp algorithm.c -lm -o  algorithm.exe

                  Note that the above compile command utilises OpenMP for parallelisation of 
                  for-loops, and assumes that 4 cores are available to process over,
                  if you want to use more than 4 cores for parallel processing you will need 
                  to change the number of cores manually in the algorithm.c code. 

    EXECUTING:--- The compile script generates algorithm.exe, which can then be executed as follows: 
    
                  ./algorithm.exe r 
                  where r tells the program to do linear regression and use linear datafile
                  
                  OR
                  
                  ./algorithm.exe a   OR   ./algorithm.exe a f [n]  
                  OR   ./algorithm.exe a c [n]   OR   ./algorithm.exe a r [n] 
                  where a tells the program to use accelerometer data
                  where a f tells the program to use specifically the fft-processed accel data
                  where a c tells the program to use specifically the cepstrogram-processed accel data
                  where a r tells the program to use running-average processed accel data
                  where the optional n tells the program to normalise the input data

    QUICKSTART:-- The following command deletes existing log files, compiles the code, runs the code, then also graphs the 
                  results, using default option of doing linear regression: 
                  source runAlgo.sh
