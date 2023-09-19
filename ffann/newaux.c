#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef MATH_H
#define MATH_H 1
#include<math.h>
#endif

#define TEST 1

//#define NULL 0

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
	Individual ** oldpopulation;
	Individual ** newpopulation;		
}Population;



int numCycles;
int nodeSizeMemory;
int individualSizeMemory;
int defaultNumberOutputNodes;

Population superpopulation;

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

//intendedResult = sys.argv[1]
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
Node * outputLayer;

//Activation functions: https://www.geeksforgeeks.org/activation-functions-neural-networks/

void initialiseVariables(){
	defaultNumberOutputNodes = 3;
	bestlms = 1000000000000000000.0; // assigning initial high value
	popsize = 4;
	hiddenMax = 20;
	hiddenMin = 5;
	outputLayerLength = 3;
	weightMax = 2.0f;
	//Set global variables values
	lmsResult = (float*) malloc(sizeof(float) * popsize);
	numCycles = 50; //global variable
	nodeSizeMemory = ( sizeof(float) * 5 ) + ( sizeof(float) * hiddenMax );
	individualSizeMemory = popsize * ( nodeSizeMemory + (hiddenMax * nodeSizeMemory) + (nodeSizeMemory * outputLayerLength) + (sizeof(float) * 5) );
	
	elitism = 1 + ( popsize / 10.0 );
	
}

float getRandomWeightValueFloat(){
	return (float) ((float)rand()/(float)(RAND_MAX)) * weightMax; //... ref:stackoverflow.com/questions/13408990 
}

float sigmoid(float value){
	return 1.0f / (1.0f + exp( (-1.0) * value) );
}

float relu(float value){
  return abs(value); // #accel data has plenty of negative values, so using absolute
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
	individualstruct->hiddenLayer = (Node **) malloc(nodeSizeMemory * individualstruct->numberOfHiddenNodes);
	individualstruct->outputLayer = (Node **) malloc(nodeSizeMemory * individualstruct->numberOfOutputNodes);
	individualstruct->best = 0.0f;
	individualstruct->lms = 4.0f;
	individualstruct->lmsLOW = 4.0f;
	individualstruct->lmsMED = 4.0f;
	individualstruct->lmsHIGH = 4.0f;
}

void constructPopulation(Population * populationstruct){
  populationstruct->oldpopulation = (Individual**) malloc( individualSizeMemory * popsize );
	populationstruct->newpopulation = (Individual**) malloc( individualSizeMemory * popsize );
}

int getRandomNumberHiddenNodesInt(){
	return (rand() % (hiddenMax - hiddenMin + 1)) + 1;
}



void process( char * filenamesList, float expectedResult, int member   ){
	int r = 0; // for choosing expected result
	int c = 0; 
	/*for(c=0; c< ; c++){
		
	}*/

}



void constructFFANN(Population* populationStruct){
	//# construct input layer, aka a single Node 
#ifdef TEST
printf("Just about to go through constructFFANN function, first to construct input node...\n");
#endif

 Node inputnode;
 constructNode(&inputnode);
 //# inputLayer = Node() 
 int numberOfHidden = getRandomNumberHiddenNodesInt();
 //#print("number of hidden: "+str(numberOfHidden))

#ifdef TEST
printf("Next to construct hidden layer....\n");
#endif

 //#construct hidden layer
 Node** hiddenLayer;
 hiddenLayer = (Node**) malloc( nodeSizeMemory * numberOfHidden );
 int c = 0;
 for(c=0; c < numberOfHidden; c++){ //#x in range(numberOfHidden):
   Node newnode;
   constructNode(&newnode);
   hiddenLayer[c] = (Node*) &newnode;
 }

 printf("length of hidden layer inside constructFFANN : %d",numberOfHidden);
 Node** outputLayer; // #3 nodes, one per expected output
 outputLayer = (Node**) malloc( nodeSizeMemory * defaultNumberOutputNodes );
 
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
			 outputLayer[c]->weights[d] = getRandomWeightValueFloat(); //#initialise weights randomly
			}
		}

 //#Now add to the current population list
 for(c=0;c<popsize;c++){
	Individual citizen;
	constructIndividual(&citizen);
	citizen.inputLayer = &inputnode;
	citizen.hiddenLayer = hiddenLayer;
	citizen.outputLayer = outputLayer;
	populationStruct->oldpopulation[c] = (Individual*) &citizen;
 }
}


