
#!/bin/python3
#https://medium.com/@b.terryjack/introduction-to-deep-learning-feed-forward-neural-networks-ffnns-a-k-a-c688d83a309d 

#run as python3 ffann.py datafilename intended result
# e.g. python3 ffan.py Accel15psi.data 15

import math
import random
import sys
import copy  #https://medium.com/python-features/cloning-objects-in-python-beginner-6ad3cd859d50
import statistics


class Node:
  def __init__(self,id):
    self.input = 0.0
    self.weight = random.random() #initialise weight randomly
    self.bias = random.random()
    self.weights = []
    self.id = id
    self.output = 0.0

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
    outputLayer.outpout = sigmoid(outputLayer.output)
    result.append( outputLayer.output )
    lms = (float(expectedResult) - outputLayer.output )
    lms = lms * lms
    lmsResult.append( lms )

global bestInputLayer
bestInputLayer = Node(100)
global bestHiddenLayer
bestHiddenLayer = []
global bestOutputLayer
bestOutputLayer = Node(101)

global inputLayer
inputLayer = Node(1)

global hiddenLayer
hiddenLayer = []
#hiddenLayer.append(Node(2))
#hiddenLayer.append(Node(3))
global outputLayer
outputLayer = Node(21)

#print(hiddenLayer[1].id)
def constructFFANN():
 #inputLayer = Node(1) 
 numberOfHidden = random.randint(2,20)
 #print("number of hidden: "+str(numberOfHidden))

 #construct hidden layer
 #hiddenLayer = []
 for x in range(numberOfHidden):
   hiddenLayer.append(Node(x))

 #print("length of hidden layer inside constructFFANN :"+str(len(hiddenLayer)))
 #outputLayer = Node(21)
 #now create output node weights, the same amount as there are hidden nodes
 for x in range( len(hiddenLayer) ):
   outputLayer.weights.append( random.random() ) #initialise weight randomly

 #print("number of hidden nodes : "+str(len(hiddenLayer)))

global result 
result = []
global lmsResult
lmsResult = []

intendedResult = sys.argv[2]

global bestlms 
bestlms= 10000000.0 # assigning initial high value

#Loop round creating a new FFANN each time to find the best one :)
for x in range(100): #100 times round the loop...100 different FFANN's

 #print("lenght of hidden layer is "+str(len(hiddenLayer)))
 process(sys.argv[1], intendedResult) # filename, func populates result list
 lmssum = sum(lmsResult)
 if lmssum < bestlms: #keep this ffann as the best so far....
     print("Found new best lms of "+str(lmssum))
     bestInputLayer = copy.deepcopy(inputLayer)
     bestHiddenLayer = copy.deepcopy(hiddenLayer)
     bestOutputLayer = copy.deepcopy(outputLayer)
 lmsresult = [] # empty this ready for the next FFANN
 #Now make new ffann....cleaning up the previous one
 inputLayer = Node(1)
 hiddenLayer = []
 outputLayer = Node(21)
 constructFFANN()

#Finally print and save the best FFANN....
print("The best FFANN for "+str(intendedResult)+" is:\n")
print("Input weight of "+str(bestInputLayer.weight)+" and bias is "+str(bestInputLayer.bias)+"\n")
for x in bestHiddenLayer:
 print("Hidden layer node"+str(x.id)+", weight is "+str(x.weight)+" and bias is "+str(x.bias)+"\n")
for x in bestOutputLayer.weights:
  print("Best output layer weight is "+str(x)+"\n ")
print("output bias is "+str(bestOutputLayer.bias))
#themean = statistics.mean(result)
#print("result mean is: "+str(themean))
#themeanlms = statistics.mean(lmsResult)
#print("mean lms is: "+str(themeanlms))
#print("best lms is: "+str(bestlms))
