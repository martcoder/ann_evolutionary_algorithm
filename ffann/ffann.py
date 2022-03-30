
#!/bin/python3
#https://medium.com/@b.terryjack/introduction-to-deep-learning-feed-forward-neural-networks-ffnns-a-k-a-c688d83a309d 
#https://en.wikipedia.org/wiki/Tournament_selection
#https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)

#run as python3 ffann.py datafilename intended result
# e.g. python3 ffan.py Accel15psi.data 15

import math
import random
import sys
import copy  #https://medium.com/python-features/cloning-objects-in-python-beginner-6ad3cd859d50
import statistics
from datetime import datetime

global popsize 
popsize = 100
global hiddenMax
hiddenMax = 40
global weightMax
weightMax = 2.0

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
    self.meanOutput = 100.0

#Activation functions: https://www.geeksforgeeks.org/activation-functions-neural-networks/
def sigmoid(value):
  return 1.0 / (1.0 + math.exp( (-1.0) * value) )

def relu(value):
  return max(0.0,value)

def process(filename,expectedResult,member):
  filehold = open(filename,"r")
  Lines = filehold.readlines()
  for x in Lines:
    splitLine = x.split(',')
    value = splitLine[0]
    inputLayer.input = float(value)

    #processing input node
    oldpopulation[member].inputLayer.output = oldpopulation[member].inputLayer.input * oldpopulation[member].inputLayer.weight # multiply input by weight
    oldpopulation[member].inputLayer.output = oldpopulation[member].inputLayer.output + oldpopulation[member].inputLayer.bias # add the bias into the mix
    oldpopulation[member].inputLayer.output = relu(oldpopulation[member].inputLayer.output) # run through activation func
    for h in oldpopulation[member].hiddenLayer:
      h.output = oldpopulation[member].inputLayer.output * h.weight
      h.output = h.output + h.bias
      h.output = relu(h.output)
    #now process the output node
    for h in range(len(oldpopulation[member].hiddenLayer)):
      #print("member number "+str(member)+" and h number "+str(h)+" and popsize is "+str(len(oldpopulation))+" and hidden len is "+str(len(oldpopulation[member].hiddenLayer))+" and weights len is "+str(len(oldpopulation[member].outputLayer.weights )))
      oldpopulation[member].outputLayer.output += oldpopulation[member].hiddenLayer[h].output * oldpopulation[member].outputLayer.weights[h]
    oldpopulation[member].outputLayer.output += oldpopulation[member].outputLayer.bias
    oldpopulation[member].outputLayer.output = relu(oldpopulation[member].outputLayer.output)
    result.append( oldpopulation[member].outputLayer.output )
    #print("result is "+str(outputLayer.output))
    lms = (float(expectedResult) - oldpopulation[member].outputLayer.output )
    lms = lms * lms
    lmsResult.append( lms )
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
  #choose which parent to get input details from
  parentInputNode = random.random()
  if(parentInputNode <= 0.5):
    newinput.weight = twoParent[0].inputLayer.weight
    newinput.bias = twoParent[0].inputLayer.bias
  else:
    newinput.weight = twoParent[1].inputLayer.weight
    newinput.bias = twoParent[1].inputLayer.bias 
  #take half of hidden from par0, half from par1
  newhidden = []
  lenP0 = len(twoParent[0].hiddenLayer)
  lenP1 = len(twoParent[1].hiddenLayer)
  newoutput = Node()
  for x in range(int(lenP0/2)):
    newhidden.append(copy.deepcopy(twoParent[0].hiddenLayer[x]) )
    newoutput.weights.append(twoParent[0].outputLayer.weights[x])
  for x in range(int(lenP1/2)):
    newhidden.append( copy.deepcopy(twoParent[1].hiddenLayer[x]) )
    newoutput.weights.append(twoParent[1].outputLayer.weights[x])
  #now truncate so not too huge....
  if len(newhidden) > 21:
    newhidden = newhidden[0:20]

  #take output based on previous prob

  #if(parentInputNode <= 0.5):
  #0  newoutput = copy.deepcopy(twoParent[0].outputLayer)
  #else:
  #  newoutput = copy.deepcopy(twoParent[1].outputLayer)
  

  #Now do random mutation
  doMutationInput = random.random()
  if(doMutationInput <0.3):
    newinput.weight = random.uniform(0.0,weightMax)
    newinput.bias = random.uniform(0.0,weightMax) #https://stackoverflow.com/questions/6088077/how-to-get-a-random-number-between-a-float-range
  doMutationHidden = random.random()
  if(doMutationHidden < 0.3):
    for x in newhidden:
      x.weight = random.uniform(0.0,weightMax)
      x.bias = random.uniform(0.0,weightMax)
  doMutationOutput = random.random()
  weightChoice = random.randint(0,len(newoutput.weights)-1)
  if doMutationOutput < 0.3:
    newoutput.weights[weightChoice] = random.uniform(0.0,weightMax)
    newoutput.bias = random.uniform(0.0,weightMax)



  #Now add the newly minted individual to the new population
  newpopulation.append(Individual(newinput,newhidden,newoutput))
  print("new populatino size is now "+str(len(newpopulation)))


 #tournament selection

#print(hiddenLayer[1].id)
def constructFFANN():
 inputLayer = Node() 
 numberOfHidden = random.randint(2,hiddenMax)
 #print("number of hidden: "+str(numberOfHidden))

 #construct hidden layer
 hiddenLayer = []
 for x in range(numberOfHidden):
   hiddenLayer.append(Node())
   hiddenLayer[x].output = 0.0

 #print("length of hidden layer inside constructFFANN :"+str(len(hiddenLayer)))
 outputLayer = Node()
 #now create output node weights, the same amount as there are hidden nodes
 outputLayer.weights = []
 for x in range( len(hiddenLayer) ):
   outputLayer.weights.append( random.random() ) #initialise weight randomly

 #Now add to the current population list
 
 oldpopulation.append(Individual(inputLayer,hiddenLayer,outputLayer))
 #print("number of hidden nodes : "+str(len(hiddenLayer)))
 inputLayer = None
 hiddenLayer = []
 outputLayer = None

global result 
result = []
global lmsResult
lmsResult = []

intendedResult = sys.argv[2]

global bestlms 
bestlms= 1000000000000000000.0 # assigning initial high value



for x in range(popsize):
  constructFFANN() # create initial population

for t in range(50): # two loops of this algorithm
  #Loop round creating a new FFANN each time to find the best one :)
  for x in range(popsize): #e.g. for each member FFANN, process it

    #print("lenght of hidden layer is "+str(len(hiddenLayer)))
    print("just about to process member "+str(x))    
    #Run through each line of data in datafile
    process(sys.argv[1], intendedResult, x) # filename, func populates result list

    #Get datafile result as LeastMeanSquared
    lmssum = sum(lmsResult)
    #print("lmssum is "+str(lmssum))
    if lmssum < bestlms: #keep this ffann as the best so far....
      print("Found new best lms of "+str(lmssum))
      meanResult = statistics.mean(result)
      print("And mean output was "+str(meanResult) )
      bestInputLayer = copy.deepcopy(oldpopulation[x].inputLayer)
      bestInputLayer.meanOutput = meanResult
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
    #lmssum = 0.0
    #Now make new ffann....cleaning up the previous one
    inputLayer = Node()
    hiddenLayer = []
    outputLayer = Node()
    #if t < 1: #only do on the first time round, as tournament will be used subsequently
    #  constructFFANN()
  
  #Now CREATE NEW POPULATION
  countElite = 0
  for x in range(popsize):
    print("about to add memer to newpop")
    if countElite < 5:
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
  writer.write("The best FFANN for "+str(intendedResult)+" with an lms of "+str(bestInputLayer.lms)+" and mean output of: "+str(bestInputLayer.meanOutput)+" is:\n")
  writer.write("Input weight of "+str(bestInputLayer.weight)+" and bias is "+str(bestInputLayer.bias)+"\n")
  for x in bestHiddenLayer:
   writer.write("Hidden layer node, weight is "+str(x.weight)+" and bias is "+str(x.bias)+"\n")
  for x in bestOutputLayer.weights:
    writer.write("Best output layer weight is "+str(x)+"\n ")
  writer.write("output bias is "+str(bestOutputLayer.bias))
  writer.close()

  print("The best FFANN for "+str(intendedResult)+" with an lms of "+str(bestInputLayer.lms)+" and mean output of: "+str(bestInputLayer.meanOutput) +" is:\n")
  print("Input weight of "+str(bestInputLayer.weight)+" and bias is "+str(bestInputLayer.bias)+"\n")
  for x in bestHiddenLayer:
   print("Hidden layer node, weight is "+str(x.weight)+" and bias is "+str(x.bias)+"\n")
  for x in bestOutputLayer.weights:
    print("Best output layer weight is "+str(x)+"\n ")
  print("output bias is "+str(bestOutputLayer.bias))
#themean = statistics.mean(result)
#print("result mean is: "+str(themean))
#themeanlms = statistics.mean(lmsResult)
#print("mean lms is: "+str(themeanlms))
#print("best lms is: "+str(bestlms))
