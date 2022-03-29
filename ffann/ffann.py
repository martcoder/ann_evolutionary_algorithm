
#!/bin/python3
#https://medium.com/@b.terryjack/introduction-to-deep-learning-feed-forward-neural-networks-ffnns-a-k-a-c688d83a309d 

#run as python3 ffann.py datafilename intended result
# e.g. python3 ffan.py Accel15psi.data 15

import math
import random
import sys
import copy  #https://medium.com/python-features/cloning-objects-in-python-beginner-6ad3cd859d50
import statistics

class Individual:
  def __init__(self,input,hidden,output):
    self.inputLayer = copy.deepcopy(input)
    self.hiddenLayer = copy.deepcopy(hidden)
    self.outputLayer = copy.deepcopy(output)
    self.best = 0

class Node:
  def __init__(self):
    self.input = 0.0
    self.weight = random.random() #initialise weight randomly
    self.bias = random.random()
    self.weights = []
    self.output = 0.0
    self.lms = 0.0

def sigmoid(value):
  return 1.0 / (1.0 + math.exp( (-1.0) * value) )

def process(filename,expectedResult):
  filehold = open(filename,"r")
  Lines = filehold.readlines()
  for x in Lines:
    splitLine = x.split(',')
    value = splitLine[0]
    inputLayer.input = float(value)

    #processing input node
    inputLayer.output = inputLayer.input * inputLayer.weight # multiply input by weight
    inputLayer.output = inputLayer.output + inputLayer.bias # add the bias into the mix
    inputLayer.output = sigmoid(inputLayer.output) # run through sigmoid activation func
    for h in hiddenLayer:
      h.output = inputLayer.output * h.weight
      h.output = h.output + h.bias
      h.output = sigmoid(h.output)
    #now process the output node
    for h in range(len(hiddenLayer)):
      outputLayer.output += hiddenLayer[h].output * outputLayer.weights[h]
    outputLayer.output += outputLayer.bias
    outputLayer.output = sigmoid(outputLayer.output)
    result.append( outputLayer.output )
    #print("result is "+str(outputLayer.output))
    lms = (float(expectedResult) - outputLayer.output )
    lms = lms * lms
    lmsResult.append( lms )
  inputLayer.lms = sum(lmsResult)

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
  for x in range(int(lenP0/2)):
    newhidden.append(copy.deepcopy(twoParent[0].hiddenLayer[x]) )
  for x in range(int(lenP1/2)):
    newhidden.append( copy.deepcopy(twoParent[1].hiddenLayer[x]) )
  #take output based on previous prob
  newoutput = Node()
  if(parentInputNode <= 0.5):
    newoutput = copy.deepcopy(twoParent[0].outputLayer)
  else:
    newoutput = copy.deepcopy(twoParent[1].outputLayer)
  

  #Now do random mutation
  doMutationInput = random.random()
  if(doMutationInput <0.3):
    newinput.weight = random.random()
    newinput.bias = random.random()
  doMutationHidden = random.random()
  if(doMutationHidden < 0.3):
    for x in newhidden:
      x.weight = random.random()
      x.bias = random.random()
  doMutationOutput = random.random()
  if doMutationOutput < 0.3:
    newoutput.weight = random.random()
    newoutput.bias = random.random()



  #Now add the newly minted individual to the new population
  newpopulation.append(Individual(newinput,newhidden,newoutput))
  print("new populatino size is now "+str(len(newpopulation)))


 #tournament selection

#print(hiddenLayer[1].id)
def constructFFANN():
 #inputLayer = Node(1) 
 numberOfHidden = random.randint(2,20)
 #print("number of hidden: "+str(numberOfHidden))

 #construct hidden layer
 #hiddenLayer = []
 for x in range(numberOfHidden):
   hiddenLayer.append(Node())
   hiddenLayer[x].output = 0.0

 #print("length of hidden layer inside constructFFANN :"+str(len(hiddenLayer)))
 #outputLayer = Node(21)
 #now create output node weights, the same amount as there are hidden nodes
 outputLayer.weights = []
 for x in range( len(hiddenLayer) ):
   outputLayer.weights.append( random.random() ) #initialise weight randomly

 #Now add to the current population list
 
 oldpopulation.append(Individual(inputLayer,hiddenLayer,outputLayer))
 #print("number of hidden nodes : "+str(len(hiddenLayer)))

global result 
result = []
global lmsResult
lmsResult = []

intendedResult = sys.argv[2]

global bestlms 
bestlms= 1000000000000000000.0 # assigning initial high value

constructFFANN() #create initial FFANN
popsize = 500

for t in range(100): # two loops of this algorithm
  #Loop round creating a new FFANN each time to find the best one :)
  for x in range(popsize): #e.g. 100 times round the loop...100 different FFANN's

    #print("lenght of hidden layer is "+str(len(hiddenLayer)))
    
    #Run through each line of data in datafile
    process(sys.argv[1], intendedResult) # filename, func populates result list

    #Get datafile result as LeastMeanSquared
    lmssum = sum(lmsResult)
    #print("lmssum is "+str(lmssum))
    if lmssum < bestlms: #keep this ffann as the best so far....
      print("Found new best lms of "+str(lmssum))
      bestInputLayer = copy.deepcopy(inputLayer)
      #print("len of hidden layer is "+str(len(hiddenLayer)))
      bestHiddenLayer = copy.deepcopy(hiddenLayer)
      bestOutputLayer = copy.deepcopy(outputLayer)
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
    if t < 1: #only do on the first time round, as tournament will be used subsequently
      constructFFANN()
  countElite = 0
  for x in range(popsize):
    if countElite < 5:
      addElite()
      countElite += 1
    else:
      tournament() #to construct new population
  oldpopulation = copy.deepcopy(newpopulation) # now copy new population to old population
  newpopulation = []
  #Finally print and save the best FFANN....
  print("current population contains "+str(len(oldpopulation))+" individuals\n")
  print("The best FFANN for "+str(intendedResult)+" with an lms of "+str(bestInputLayer.lms)+" is:\n")
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
