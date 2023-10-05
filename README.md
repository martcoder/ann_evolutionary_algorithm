# ann_evolutionary_algorithm

===TITLE=========================

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

Some python scripts are also present as they were useful in automatically generating hundreds of struct 
declarations in the C code file (as C doesn't have dynamic arrays, the approach taken was to declare the
maximum amount of structs needed, then use up to that maximum when executing). 


===COMPILING AND RUNNING=========

    ENVIRONMENT:- The project was built on linux for ease of access to a C compiler. 

    COMPILING:--- Included in the repository is a compile.sh shell script, which can be ran using: 
                  source compile.sh
    EXECUTING:--- The compile script generates new.exe, which can then be executed as follows: 
                  ./new.exe a f
                  where a tells the program to use accelerometer data
                  where f tells the program to use specifically the fft-processed accel data
                  (including another letter in place of f will therefore use the raw accel data)
