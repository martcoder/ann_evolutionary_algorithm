#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

//https://medium.com/@b.terryjack/introduction-to-deep-learning-feed-forward-neural-networks-ffnns-a-k-a-c688d83a309d 
//https://en.wikipedia.org/wiki/Tournament_selection
//https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)

//Before running, ensure you edit script to include the data filenames you want to be processed
//... they are just a few lines down from here....
//run as python3 ffann.py accel | lidar
// e.g. python3 ffan.py accel
//EDIT: The expected results are now hard-coded
//... into the process function! So if you want to change them they needed editing there

typedef struct node {
	float input;
	float weight; // initialise using ((float)rand()/(float)(RAND_MAX)) * upperLimit ... ref:stackoverflow.com/questions/13408990 
	float bias;
	float * weights;
	float output;
	float lms; //initialse to highh value
}Node;

typedef struct individual {
	Node * inputLayer;
	int numberOfHiddenNodes;
	Node * hiddenLayer;
	int numberOfOutputNodes;
	Node * outputLayer;
	float best;
	float lms;
	float lmsLOW;
	float lmsMED;
	float lmsHIGH;
}Individual;

typedef struct population {
	Individual * oldpopulation;
	Individual * newpopulation;		
}Population;

int numCycles;
int nodeSizeMemory;
int individualSizeMemory;
int defaultNumberOutputNodes;

Population population;

char * filenamesList[3][1024]; //3 arrays of strings
char filenamesListLow[3][1024]; //3 strings
char filenamesListMiddle[4][1024]; //4 strings
char filenamesListHigh[3][1024]; //3 strings
int chosenSensor; //0 is for accelerometer, 1 is for lidar

/*
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
global LMSresultLOW
LMSresultLOW = []
global LMSresultMED
LMSresultMED = []
global LMSresultHIGH
LMSresultHIGH = []
*/

float * lmsResult;

intendedResult = sys.argv[1]
float bestlms; 
//bestlms= 1000000000000000000.0 # assigning initial high value


int popsize;
//popsize = 4
int hiddenMax;
//hiddenMax = 20
int hiddenMin; 
int outputLayerLength; 
float weightMax;
//weightMax = 2.0
float elitism;
//elitism = max(1, math.ceil( popsize / 10.0 ) )

//Set useful variables which define structure of ANNs and hold final best coefficients
Node bestInputLayer;
//bestInputLayer = Node()
Node * bestHiddenLayer;
//bestHiddenLayer = []
Node * bestOutputLayer;

Node inputLayer;

Node * hiddenLayer;

//hiddenLayer = []
//#hiddenLayer.append(Node(2))
//#hiddenLayer.append(Node(3))
Node * outputLayer

//Activation functions: https://www.geeksforgeeks.org/activation-functions-neural-networks/

void initialiseVariables(){
	defaultNumberOutputNodes = 3;
	bestlms = 1000000000000000000.0 // assigning initial high value
	popsize = 4;
	hiddenMax = 20;
	hiddenMin = 5;
	outputLayerLength = 3;
	weightMax = 2.0f;
	//Set global variables values
	lmsResult = (float*) malloc(sizeof(float) * popsize);
	numCycles = 50 //global variable
	nodeSizeMemory = ( sizeof(float) * 5 ) + ( sizeof(float) * hiddenMax );
	individualSizeMemory = popsize * ( nodeSizeMemory + (hiddenMax * nodeSizeMemory) + (nodeSizeMemory * outputLength) + (sizeof(float) * 5) );
	
	elitism = 1 + ( popsize / 10.0 );
	
}

float sigmoid(float value){
	return 1.0f / (1.0f + exp( (-1.0) * value) )
}

float relu(float value){
  return abs(value) // #accel data has plenty of negative values, so using absolute
}

void process( char * filenamesList, float expectedResult, int member   ){
	r = 0; // for choosing expected result
	int c = 0; 
	for(c=0; c< ; c++){
		
	}

}
def process(filenamesList,expectedResult,member):
 global global_population #need this global keyword to access global object !!!
 r = 0 #for choosing the expected result, will increment at the end of the following for loop
 for filenamearray in filenamesList:
  if r == 0:
    expectedResult = 5100 # LOW pressure data being trained with
  if r == 1:
    expectedResult = 5010 # MED pressure data being trained with
  if r == 2:
    expectedResult = 5001 # HIGH pressure data being trained with

  for name in filenamearray:
   #print('processing datafile '+name)
   filehold = open(name,"r")
   Lines = filehold.readlines()
   for x in Lines:
     splitLine = x.split(',')
     value = splitLine[0]
     floatval = float(value)
     if chosenSensor == 'lidar':
       if (floatval) < 400.0 or (floatval > 650.0): #filter outliers
          floatval = 500.0 #Remove outlier and just use regular value
     #print("Population member number is "+str(member))
     #print("Population input weight is "+str(oldpopulation[member].inputLayer.weight))
     #print("Input value stored in this member is currently "+str(oldpopulation[member].inputLayer.input) )
     global_population.oldpopulation[member].inputLayer.input = floatval

     #processing input node
     global_population.oldpopulation[member].inputLayer.output = global_population.oldpopulation[member].inputLayer.input * global_population.oldpopulation[member].inputLayer.weight # multiply input by weight
     global_population.oldpopulation[member].inputLayer.output = global_population.oldpopulation[member].inputLayer.output + global_population.oldpopulation[member].inputLayer.bias # add the bias into the mix
     global_population.oldpopulation[member].inputLayer.output = relu(global_population.oldpopulation[member].inputLayer.output) # run through activation func
     for h in global_population.oldpopulation[member].hiddenLayer:
       h.output = global_population.oldpopulation[member].inputLayer.output * h.weight
       h.output = h.output + h.bias
       h.output = relu(h.output)
     #now process the output node
     #oldpopulation[member].outputLayer.output = 0.0
     normalisingList = []
     for oi in range(len(global_population.oldpopulation[member].outputLayer)):
        global_population.oldpopulation[member].outputLayer[oi].output = 0.0 #first set to zero for fresh numbers
        for h in range(len(global_population.oldpopulation[member].hiddenLayer)):
	  #print("member number "+str(member)+" and h number "+str(h)+" and popsize is "+str(len(oldpopulation))+" and hidden len is "+str(len(oldpopulation[member].hiddenLayer))+" and weights len is "+str(len(oldpopulation[member].outputLayer.weights )))
           global_population.oldpopulation[member].outputLayer[oi].output += global_population.oldpopulation[member].hiddenLayer[h].output * global_population.oldpopulation[member].outputLayer[oi].weights[h]
        global_population.oldpopulation[member].outputLayer[oi].output += global_population.oldpopulation[member].outputLayer[oi].bias
        global_population.oldpopulation[member].outputLayer[oi].output = relu(global_population.oldpopulation[member].outputLayer[oi].output)
        #print("Final output is "+str(global_population.oldpopulation[member].outputLayer[oi].output))
        normalisingList.append( global_population.oldpopulation[member].outputLayer[oi].output ) 
     #print("result is "+str(outputLayer.output))
     try:
       largestOutput = max(normalisingList)
     except ValueError:
       print("normalisingList seems to not be populated")
       print("first output node value is "+str(global_population.oldpopulation[0].outputLayer[0].output)) 
     #print("largest output is "+str(largestOutput))
     indexOfLargestOutput = normalisingList.index(max(normalisingList))

      #Populise a normalised version of the outputs...
     normalisedOutput = []
     normalisedOutput.clear()
     secondLargestIndex = 0
     for n in range(len(normalisingList)):
       normalisedOutput.append( normalisingList[n] / largestOutput )
       #print("NormalisedOutput "+str(n)+":"+str(normalisedOutput[n]))
       if( (normalisingList[n] < largestOutput) and (normalisingList[n] > normalisingList[secondLargestIndex]) ):
         secondLargestIndex = n #update which index contains the 2nd largest number

     
     #resultLOW.append( oldpopulation[member].outputLayer[0].output )
     #expectedResultMED = 40.0
     #resultMED.append( oldpopulation[member].outputLayer[1].output )

     #expectedResultHIGH = 60.0
     #resultHIGH.append( oldpopulation[member].outputLayer[2].output  )
     #LMSresultLOW.append( (float(expectedResult) - oldpopulation[member].outputLayer.output )  )
     #LMSresultMED.append(   )

     if expectedResult == 5001: #HIGH PRESSURE
        if(indexOfLargestOutput != 2): #Nowhere near where want to be, so....
          lms = 2 #automatic penalty
        else: # This is good news, so  action this....
          # by minusing 1 - 2ndLargest this will produce large LMS when 2ndlargest is large, and small LMS when 2ndlargest is small,
          # and as small LMS is better, this will punish a large 2ndlargest value :D
          lms = 1 - (1 - normalisedOutput[secondLargestIndex] )
        #print("Normalised output values for HIGH pressure data are LOW-MED-HIGH: "+str(normalisedOutput[0])+"-"+str(normalisedOutput[1])+"-"+str(normalisedOutput[0]))
     elif expectedResult == 5010: #MED PRESSURE
        if indexOfLargestOutput != 1:
          lms = 2 # automatic punishment
        else: 
          lms = 1 - (1 - normalisedOutput[secondLargestIndex] )
        #print("Normalised output values for MED pressure data are LOW-MED-HIGH: "+str(normalisedOutput[0])+"-"+str(normalisedOutput[1])+"-"+str(normalisedOutput[0]))
     elif expectedResult == 5100: #LOW PRESSURE
       if indexOfLargestOutput != 0:
          lms = 2
       else:
          lms = 1 - (1 - normalisedOutput[secondLargestIndex] ) 
       #print("Normalised output values for LOW pressure data are LOW-MED-HIGH: "+str(normalisedOutput[0])+"-"+str(normalisedOutput[1])+"-"+str(normalisedOutput[0]))
     else:
       print("Hmmm the expected result was not one of the usual ones...")

     #lms = (float(expectedResult) - oldpopulation[member].outputLayer.output )
     lms = lms * lms
     lmsResult.append( lms )
  r += 1 #increment which expected result needs using
 print("Member "+str(member)+" has processed low med and high data and has avg lms of "+str(statistics.mean(lmsResult)))
 global_population.oldpopulation[member].inputLayer.lms = statistics.mean(lmsResult)


def addElite():
  global_population.newpopulation.append(Individual(bestInputLayer,bestHiddenLayer,bestOutputLayer))


def tournament():
  tournySet = []
  # Select 4 random individuals. 
  indv = random.randint(0,len(global_population.oldpopulation)-1)
  
  #print("old pop size is "+str(len(oldpopulation))+" and index chosen "+str(indv))
  tournySet.append( global_population.oldpopulation[indv]  )
  indv = random.randint(1,len(global_population.oldpopulation)-1)
  #print("old pop size is "+str(len(oldpopulation))+" and index chosen "+str(indv))

  tournySet.append( global_population.oldpopulation[indv]  )
  indv = random.randint(1,len(global_population.oldpopulation)-1)
  #print("old pop size is "+str(len(oldpopulation))+" and index chosen "+str(indv))

  tournySet.append( global_population.oldpopulation[indv]  )
  indv = random.randint(1,len(global_population.oldpopulation)-1)
  #print("old pop size is "+str(len(oldpopulation))+" and index chosen "+str(indv))

  tournySet.append( global_population.oldpopulation[indv]  )

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
  newoutput = []
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

    newoutput.append( Node() )
    newoutput.append( Node() )
    newoutput.append( Node() )

    for x in range(int(math.ceil(lenP0/2))): #cycle through 1/2 parent
      newhidden.append(copy.deepcopy(twoParent[0].hiddenLayer[x]) )
    newoutput[0].weights = twoParent[0].outputLayer[0].weights 
      #for w in range(len(twoParent[0].outputLayer)): #cycle through 3 outputs
      #  newoutputLayer[w].weights.append( twoParent[0].outputLayer[w].weights[x] )
    for x in range(int(math.ceil(lenP1/2))): #cycle through 1/2 parent
      newhidden.append( copy.deepcopy(twoParent[1].hiddenLayer[x]) )
    newoutput[1].weights = twoParent[1].outputLayer[1].weights
    newoutput[2].weights = twoParent[1].outputLayer[2].weights
      #for w in range(len(twoParent[1].outputLayer)): #cycle through 3 outputs
      #  newoutputLayer[w].weights.append(twoParent[0].outputLayer[w].weights[x])

    #now truncate so not too huge....

    if len(newhidden) > (hiddenMax+1):
       newhidden = newhidden[0:hiddenMax]
    print("newhidden length is "+str(len(newhidden)))
    for x in range(len(newoutput)):
      if len(newoutput[x].weights) > (hiddenMax+1):
         newoutput[x].weights = newoutput[x].weights[0:hiddenMax] 
    #take output bias based on previous prob
    if(parentInputNode <= 0.5):
       for x in range(3):
         newoutput[x].bias = twoParent[0].outputLayer[x].bias
    else:
       for x in range(3):
         newoutput[x].bias = twoParent[1].outputLayer[x].bias
  

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
  #weightChoice = random.randint(0,len(newoutput[0].weights)-1) #arbitrary choice of first output node for weights length
  #print("weight number for output node is "+str(weightChoice))
  if doMutationOutput < 0.3:
    for x in range(len(newoutput)):
      weightChoice = random.randint(0,len(newoutput[x].weights)-1)
      newoutput[x].weights[weightChoice] = random.uniform(0.0,weightMax) #mutate that chosen weight
      newoutput[x].bias = random.uniform(0.0,weightMax) #also mutate bias



  #Now add the newly minted individual to the new population
  global_population.newpopulation.append(Individual(newinput,newhidden,newoutput))
  print("new populatino size is in tournament now "+str(len(global_population.newpopulation)))



void constructFFANN(Population* populationStruct){
	//# construct input layer, aka a single Node 
 Node inputnode;
 constructNode(&inputnode);
 //# inputLayer = Node() 
 int numberOfHidden = getRandomNumberHiddenNodesInt();
 //#print("number of hidden: "+str(numberOfHidden))

 //#construct hidden layer
 Node* hiddenLayer;
 hiddenLayer = (Node*) malloc( nodeSizeMemory * numberOfHidden );
 int c = 0;
 for(c=0; c < numberOfHidden; c++){ //#x in range(numberOfHidden):
   Node newnode;
   constructNode(&newnode);
   hiddenLayer[c] = &newnode;
 }

 printf("length of hidden layer inside constructFFANN : %d",numberOfHidden);
 Node* outputLayer; // #3 nodes, one per expected output
 outputLayer = (Node*) malloc( nodeSizeMemory * defaultNumberOutputNodes );
 
	Node outputNode0;
	Node outputNode1;
	Node outputNode2;
	constructNode( &outputNode0 );
	outputLayer[c] = &outputNode0;
	constructNode( &outputNode1 );
	outputLayer[c] = &outputNode1;
	constructNode( &outputNode2 );
	outputLayer[c] = &outputNode2;
 
 //#now create output node weights, the same amount as there are hidden nodes
 for(c=0; c < defaultNumberOutputNodes; c++){
   int d = 0;
   for(d=0; d < numberOfHidden; d++){ 
		 outputLayer[c]->weights[d] = 0.0f; //#clear the weights
		 //#need same number of weights in each output node as there are hidden nodes...
			 outputLayer[c]->weights[d] = getRandomWeightValueFloat(); #initialise weights randomly
			}
		}

 #Now add to the current population list
 for(c=0;c<popsize;c++){
	Individual citizen;
	constructIndividual(&citizen);
	citizen->inputLayer) = &inputnode;
	citizen->hiddenLayer = hiddenLayer;
	citizen->outputLayer = outputLayer;
	populationStruct->oldpopulation[c] = &citizen;
 }
}

void constructNode(Node * nodestruct){
	nodestruct->input = 0.0f;
	nodestruct->weight = getRandomWeightValueFloat();
	nodestruct->bias = 0.0f;
	nodestruct->weights = (float *) malloc( sizeof(float) * hiddenMax );
	nodestruct->output = 0.0f;
	nodestruct->lms = 2.0f; // initialising to a high value...
}

void constructIndividual(Individual * individualstruct){
  individualstruct->numberOfHiddenNodes = getRandomNumberHiddenNodesInt();
  individualstruct->numberOfOutputNodes = defaultNumberOutputNodes; 
	individualstruct->inputLayer = (Node *) malloc(nodeSizeMemory);
	individualstruct->hiddenLayer = (Node *) malloc(nodeSizeMemory * individualstruct->numberOfHiddenNodes);
	individualstruct->outputLayer = (Node *) malloc(nodeSizeMemory * individualstruct->numberOfOutputNodes);
	individualstruct->best = 0.0f;
	individualstruct->lms = 4.0f;
	individualstruct->lmsLOW = 4.0f;
	individualstruct->lmsMED = 4.0f;
	individualstruct->lmsHIGH = 4.0f;
}

void constructPopulation(Population * populationstruct){
  populationstruct->oldpopulation = (Individual*) malloc( individualSizeMemory * popsize );
	populationstruct->newpopulation = (Individual*) malloc( individualSizeMemory * popsize );
}

int getRandomNumberHiddenNodesInt(){
	return (rand() % (hiddenMax - hiddenMin + 1)) + 1;
}

float getRandomWeightValueFloat(){
	return ((float)rand()/(float)(RAND_MAX)) * weightMax; //... ref:stackoverflow.com/questions/13408990 
}

