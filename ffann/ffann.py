
#!/bin/python3
#https://medium.com/@b.terryjack/introduction-to-deep-learning-feed-forward-neural-networks-ffnns-a-k-a-c688d83a309d 
#https://en.wikipedia.org/wiki/Tournament_selection
#https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)

#Before running, ensure you edit script to include the data filenames you want to be processed
#... they are just a few lines down from here....
#run as python3 ffann.py accel | lidar
# e.g. python3 ffan.py accel
#EDIT: The expected results are now hard-coded
#... into the process function! So if you want to change them they needed editing there

import math
import random
import sys
import copy  #https://medium.com/python-features/cloning-objects-in-python-beginner-6ad3cd859d50
import statistics
from datetime import datetime

global filenamesList
global filenamesListLow
global filenamesListMiddle
global filenamesListHigh
global chosenSensor
if sys.argv[1] == 'accel':
  chosenSensor = 'accel'
  filenamesListLow = ['Accel15psi.data','Accel20psi.data','Accel25psi.data']
  filenamesListMiddle = ['Accel30psi.data','Accel35psi.data','Accel40psi.data','Accel45psi.data']
  filenamesListHigh = ['Accel50psi.data','Accel55psi.data','Accel60psi.data']
elif sys.argv[1] == 'lidar':
  chosenSensor = 'lidar'
  filenamesListLow = ['Lidar15psi.data','Lidar20psi.data','Lidar25psi.data']
  filenamesListMiddle = ['Lidar30psi.data','Lidar35psi.data','Lidar40psi.data','Lidar45psi.data']
  filenamesListHigh = ['Lidar50psi.data','Lidar55psi.data','Lidar60psi.data']
else:
  print('You need to specify accel or lidar as the argument when running this script')
  exit()

filenamesList = [filenamesListLow,filenamesListMiddle,filenamesListHigh]
global popsize 
popsize = 400
global hiddenMax
hiddenMax = 20
hiddenMin = 5
global weightMax
weightMax = 2.0
global elitism
elitism = 20

class Individual:
  def __init__(self,input,hidden,output):
    self.inputLayer = copy.deepcopy(input)
    self.hiddenLayer = copy.deepcopy(hidden)
    self.outputLayer = copy.deepcopy(output)
    self.best = 0

class Node:
  def __init__(self):
    self.input = 0.0
    self.weight = random.uniform(0.0,weightMax) #initialise weight randomly
    self.bias = random.uniform(0.0,weightMax)
    self.weights = []
    self.output = 0.0
    self.lms = 999999999999999.0
    self.meanOutputLOW = 100.0
    self.meanOutputMED = 100.0
    self.meanOutputHIGH = 100.0

#Activation functions: https://www.geeksforgeeks.org/activation-functions-neural-networks/
def sigmoid(value):
  return 1.0 / (1.0 + math.exp( (-1.0) * value) )

def relu(value):
  return max(0.0,value)

def process(filenamesList,expectedResult,member):
 r = 0 #for choosing the expected result
 for filenamearray in filenamesList:
  for name in filenamearray:
   #print('processing datafile '+name)
   filehold = open(name,"r")
   Lines = filehold.readlines()
   for x in Lines:
     splitLine = x.split(',')
     value = splitLine[0]
     floatval = float(value)
     if chosenSensor == 'lidar':
       if (floatval) < 400 or (floatval > 650): #filter outliers
          floatval = 500.0 #Remove outlier and just use regular value
     inputLayer.input = floatval

     #processing input node
     oldpopulation[member].inputLayer.output = oldpopulation[member].inputLayer.input * oldpopulation[member].inputLayer.weight # multiply input by weight
     oldpopulation[member].inputLayer.output = oldpopulation[member].inputLayer.output + oldpopulation[member].inputLayer.bias # add the bias into the mix
     oldpopulation[member].inputLayer.output = relu(oldpopulation[member].inputLayer.output) # run through activation func
     for h in oldpopulation[member].hiddenLayer:
       h.output = oldpopulation[member].inputLayer.output * h.weight
       h.output = h.output + h.bias
       h.output = relu(h.output)
     #now process the output node
     oldpopulation[member].outputLayer.output = 0.0
     for h in range(len(oldpopulation[member].hiddenLayer)):
       #print("member number "+str(member)+" and h number "+str(h)+" and popsize is "+str(len(oldpopulation))+" and hidden len is "+str(len(oldpopulation[member].hiddenLayer))+" and weights len is "+str(len(oldpopulation[member].outputLayer.weights )))
       oldpopulation[member].outputLayer.output += oldpopulation[member].hiddenLayer[h].output * oldpopulation[member].outputLayer.weights[h]
     oldpopulation[member].outputLayer.output += oldpopulation[member].outputLayer.bias
     oldpopulation[member].outputLayer.output = relu(oldpopulation[member].outputLayer.output)
     result.append( oldpopulation[member].outputLayer.output )
     #print("result is "+str(outputLayer.output))
     if r == 0:
      expectedResult = 200000.0
      resultLOW.append( oldpopulation[member].outputLayer.output )
     elif r == 1:
      expectedResult = 400000.0
      resultMED.append( oldpopulation[member].outputLayer.output )

     elif r == 2:
      expectedResult = 600000.0
      resultHIGH.append( oldpopulation[member].outputLayer.output  )
     else:
      pass
     lms = (float(expectedResult) - oldpopulation[member].outputLayer.output )
     lms = lms * lms
     lmsResult.append( lms )
  r += 1 #increment which expected result needs using

 oldpopulation[member].inputLayer.lms = sum(lmsResult)

global bestInputLayer
bestInputLayer = Node()
global bestHiddenLayer
bestHiddenLayer = []
global bestOutputLayer
bestOutputLayer = Node()

global inputLayer
inputLayer = Node()

global hiddenLayer
hiddenLayer = []
#hiddenLayer.append(Node(2))
#hiddenLayer.append(Node(3))
global outputLayer
outputLayer = Node()

global oldpopulation
oldpopulation = []
global newpopulation
newpopulation = []


def addElite():
  newpopulation.append(Individual(bestInputLayer,bestHiddenLayer,bestOutputLayer))


def tournament():
  tournySet = []
  # Select 4 random individuals. 
  indv = random.randint(0,len(oldpopulation)-1)
  
  #print("old pop size is "+str(len(oldpopulation))+" and index chosen "+str(indv))
  tournySet.append( oldpopulation[indv]  )
  indv = random.randint(1,len(oldpopulation)-1)
  #print("old pop size is "+str(len(oldpopulation))+" and index chosen "+str(indv))

  tournySet.append( oldpopulation[indv]  )
  indv = random.randint(1,len(oldpopulation)-1)
  #print("old pop size is "+str(len(oldpopulation))+" and index chosen "+str(indv))

  tournySet.append( oldpopulation[indv]  )
  indv = random.randint(1,len(oldpopulation)-1)
  #print("old pop size is "+str(len(oldpopulation))+" and index chosen "+str(indv))

  tournySet.append( oldpopulation[indv]  )

  #choose 2 best to parent
  twoParent = []
  twoParent.append(tournySet[0])
  twoParent.append(tournySet[1])
  if tournySet[2].inputLayer.lms < twoParent[0].inputLayer.lms:
    twoParent[0] = tournySet[2]
  elif tournySet[2].inputLayer.lms < twoParent[1].inputLayer.lms:
    twoParent[1] = tournySet[2]
  if tournySet[3].inputLayer.lms < twoParent[0].inputLayer.lms:
    twoParent[0] = tournySet[3]
  elif tournySet[3].inputLayer.lms < twoParent[1].inputLayer.lms:
    twoParent[1] = tournySet[3]

  #now do breeding
  newinput = Node()
  newhidden = []
  newoutput = Node()
  #choose whether to crossover
  crossover = random.random()
  if crossover > 0.9:
   newinput = copy.deepcopy(twoParent[0].inputLayer) # just keep a good parent
   newhidden = copy.deepcopy(twoParent[0].hiddenLayer)
   newoutput = copy.deepcopy(twoParent[0].outputLayer)
  else: #if crossover <= 0.9 then DO CROSSOVER, so the majority of the time

    #choose which parent to get input details from
    parentInputNode = random.random()
    if(parentInputNode <= 0.5):
      newinput.weight = twoParent[0].inputLayer.weight
      newinput.bias = twoParent[0].inputLayer.bias
    else:
      newinput.weight = twoParent[1].inputLayer.weight
      newinput.bias = twoParent[1].inputLayer.bias 
    #take half of hidden from par0, half from par1
    #newhidden = []
    lenP0 = len(twoParent[0].hiddenLayer)
    lenP1 = len(twoParent[1].hiddenLayer)
    #newoutput = Node()
    for x in range(int(math.ceil(lenP0/2))): #cycle through 1/2 parent
      newhidden.append(copy.deepcopy(twoParent[0].hiddenLayer[x]) )
      newoutput.weights.append(twoParent[0].outputLayer.weights[x] )
      #for w in range(len(twoParent[0].outputLayer)): #cycle through 3 outputs
      #  newoutputLayer[w].weights.append( twoParent[0].outputLayer[w].weights[x] )
    for x in range(int(math.ceil(lenP1/2))): #cycle through 1/2 parent
      newhidden.append( copy.deepcopy(twoParent[1].hiddenLayer[x]) )
      newoutput.weights.append( twoParent[1].outputLayer.weights[x]  )
      #for w in range(len(twoParent[1].outputLayer)): #cycle through 3 outputs
      #  newoutputLayer[w].weights.append(twoParent[0].outputLayer[w].weights[x])

    #now truncate so not too huge....
    if len(newhidden) > (hiddenMax+1):
       newhidden = newhidden[0:hiddenMax]
    #for x in range(len(newoutput)):
    if len(newoutput.weights) > (hiddenMax+1):
       newoutput.weights = newoutput.weights[0:hiddenMax] 
    #take output bias based on previous prob
    if(parentInputNode <= 0.5):
       newoutput.bias = twoParent[0].outputLayer.bias
    else:
       newoutput.bias = twoParent[1].outputLayer.bias
  

  #Now do random mutation
  doMutationInput = random.random()
  if(doMutationInput < 0.3):
    newinput.weight = random.uniform(0.0,weightMax)
    newinput.bias = random.uniform(0.0,weightMax) #https://stackoverflow.com/questions/6088077/how-to-get-a-random-number-between-a-float-range
  doMutationHidden = random.random()
  if(doMutationHidden < 0.3):
    for x in newhidden:
      x.weight = random.uniform(0.0,weightMax)
      x.bias = random.uniform(0.0,weightMax)
  doMutationOutput = random.random()
  #choose a random weight index
  weightChoice = random.randint(0,len(newoutput.weights)-1) #arbitrary choice of first output node for weights length
  if doMutationOutput < 0.3:
    #for x in newoutput:
      newoutput.weights[weightChoice] = random.uniform(0.0,weightMax) #mutate that chosen weight
      newoutput.bias = random.uniform(0.0,weightMax) #also mutate bias



  #Now add the newly minted individual to the new population
  newpopulation.append(Individual(newinput,newhidden,newoutput))
  print("new populatino size is now "+str(len(newpopulation)))


 #tournament selection

#print(hiddenLayer[1].id)
def constructFFANN():
 inputLayer = Node() 
 numberOfHidden = random.randint(hiddenMin,hiddenMax)
 #print("number of hidden: "+str(numberOfHidden))

 #construct hidden layer
 hiddenLayer = []
 for x in range(numberOfHidden):
   hiddenLayer.append(Node())
   hiddenLayer[x].output = 0.0

 #print("length of hidden layer inside constructFFANN :"+str(len(hiddenLayer)))
 outputLayer = Node() #[] #3 nodes, one per expected output
 #outputLayer.append( Node() )
 #outputLayer.append( Node() )
 #outputLayer.append( Node() )
 #now create output node weights, the same amount as there are hidden nodes
 #for o in outputLayer:
 # o.weights = []
 for x in range( len(hiddenLayer) ):
    outputLayer.weights.append( random.random() ) #initialise weights randomly

 #Now add to the current population list
 
 oldpopulation.append(Individual(inputLayer,hiddenLayer,outputLayer))
 #print("number of hidden nodes : "+str(len(hiddenLayer)))
 inputLayer = None
 hiddenLayer = []
 outputLayer = None

global result 
result = []
global resultLOW
resultLOW = []
global resultMED
resultMED = []
global resultHIGH
resultHIGH = []
global lmsResult
lmsResult = []

intendedResult = sys.argv[1]

global bestlms 
bestlms= 1000000000000000000.0 # assigning initial high value



for x in range(popsize):
  constructFFANN() # create initial population

for t in range(100): # number of cycles of this evolutionary algorithm
  #Process the input data through each population member
  for x in range(popsize): #e.g. for each member FFANN, process it

    #print("lenght of hidden layer is "+str(len(hiddenLayer)))
    print("cycle is "+str(t)+", just about to process member "+str(x))    
    #Run through each line of data in datafile
    process(filenamesList, intendedResult, x) # filename, func populates result list

    #Get datafile result as LeastMeanSquared
    lmssum = sum(lmsResult)
    #print("lmssum is "+str(lmssum))
    if lmssum < bestlms: #Set this ffann as the best so far....
      print("Found new best lms of "+str(lmssum))
      meanResult = statistics.mean(result)
      print("And mean output was "+str(meanResult) )
      meanLOW = statistics.mean(resultLOW)
      meanMED = statistics.mean(resultMED)
      meanHIGH = statistics.mean(resultHIGH)
      print("mean LOW was "+str(meanLOW)+", mean MED was "+str(meanMED)+", mean HIGH was "+str(meanHIGH))
      bestInputLayer = copy.deepcopy(oldpopulation[x].inputLayer)
      bestInputLayer.meanOutput = meanResult
      bestInputLayer.meanOutputLOW = meanLOW
      bestInputLayer.meanOutputMED = meanMED
      bestInputLayer.meanOutputHIGH = meanHIGH
      #print("len of hidden layer is "+str(len(hiddenLayer)))
      bestHiddenLayer = copy.deepcopy(oldpopulation[x].hiddenLayer)
      bestOutputLayer = copy.deepcopy(oldpopulation[x].outputLayer)
      #print("The best FFANN for "+str(intendedResult)+" is:\n")
      #print("Input weight of "+str(bestInputLayer.weight)+" and bias is "+str(bestInputLayer.bias)+"\n")
      #for x in bestHiddenLayer:
      #  print("Hidden layer node"+str(x.id)+", weight is "+str(x.weight)+" and bias is "+str(x.bias)+"\n")
      #for x in bestOutputLayer.weights:
      #  print("Best output layer weight is "+str(x)+"\n ")
      #print("output bias is "+str(bestOutputLayer.bias))

      bestlms = lmssum
    lmsResult = [] # empty this ready for the next FFANN
    result = []
    resultMED = []
    resultLOW = []
    resultHIGH = []
    #lmssum = 0.0
    #Now make new ffann....cleaning up the previous one
    inputLayer = Node()
    hiddenLayer = []
    outputLayer = Node()
    #if t < 1: #only do on the first time round, as tournament will be used subsequently
    #  constructFFANN()
  
  #Now CREATE NEW POPULATION
  #Firstly do elitism
  countElite = 0
  for x in range(popsize):
    print("about to add memer to newpop")
    if countElite < elitism: #elitism num defined at beginning of script
      addElite()
      countElite += 1
    else:
      if len(newpopulation) < popsize:
        print("about to tournament")
        tournament() #to construct new population member
  countElite = 0
  oldpopulation = []
  oldpopulation = copy.deepcopy(newpopulation) # now copy new population to old population
  newpopulation = []
  #Finally print and save the best FFANN....
  print("current population contains "+str(len(oldpopulation))+" individuals\n")
  
  writer = open(str(datetime.now())+"_"+str(intendedResult)+".log","a")
  writer.write("The best FFANN for "+str(intendedResult)+" with an lms of "+str(bestInputLayer.lms)+" and mean output of: "+str(bestInputLayer.meanOutput)+", and meanLOW of "+str(bestInputLayer.meanOutputLOW)+", and meanMED: "+str(bestInputLayer.meanOutputMED)+", and meanHIGH: "+str(bestInputLayer.meanOutputHIGH)+" is:\n")
  writer.write("Input weight of "+str(bestInputLayer.weight)+" and bias is "+str(bestInputLayer.bias)+"\n")
  for x in bestHiddenLayer:
   writer.write("Hidden layer node, weight is "+str(x.weight)+" and bias is "+str(x.bias)+"\n")
  #for m in bestOutputLayer:
  for x in bestOutputLayer.weights:
     writer.write("Best output layer weight is "+str(x)+"\n ")
  writer.write("output bias is "+str(bestOutputLayer.bias))
  writer.close()

  print("The best FFANN for "+str(intendedResult)+" with an lms of "+str(bestInputLayer.lms)+" and mean output of: "+str(bestInputLayer.meanOutput) +", and meanLOW of "+str(bestInputLayer.meanOutputLOW)+", and meanMED: "+str(bestInputLayer.meanOutputMED)+", and meanHIGH: "+str(bestInputLayer.meanOutputHIGH)+" is:\n")
  print("Input weight of "+str(bestInputLayer.weight)+" and bias is "+str(bestInputLayer.bias)+"\n")
  for x in bestHiddenLayer:
   print("Hidden layer node, weight is "+str(x.weight)+" and bias is "+str(x.bias)+"\n")
  
  #for e in bestOutputLayer:
  for x in bestOutputLayer.weights:
    print("Best output layer weight is "+str(x)+"\n ")
  print("output bias is "+str(bestOutputLayer.bias))
#themean = statistics.mean(result)
#print("result mean is: "+str(themean))
#themeanlms = statistics.mean(lmsResult)
#print("mean lms is: "+str(themeanlms))
#print("best lms is: "+str(bestlms))
