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
	Node ** hiddenLayer;
	int numberOfOutputNodes;
	Node ** outputLayer;
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


Individual i0;
Individual i1;
Individual i2;
Individual i3;
Individual i4;
Individual i5;
Individual i6;
Individual i7;
Individual i8;
Individual i9;
Individual i10;
Individual i11;
Individual i12;
Individual i13;
Individual i14;
Individual i15;
Individual i16;
Individual i17;
Individual i18;
Individual i19;


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

float getRandomBiasValueFloat(float max, float min){
	return (float) (((float)rand()/(float)(RAND_MAX)) * (max - min)) + min; //... ref:stackoverflow.com/questions/13408990 
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
	nodestruct->bias = getRandomBiasValueFloat(weightMax*2.0, weightMax*-2.0);
	nodestruct->weights = (float *) malloc( sizeof(float) * hiddenMax );
	nodestruct->output = 0.0f;
	nodestruct->lms = 2.0f; // initialising to a high value...
}

void constructIndividual(Individual * individualstruct, int paramNumberHiddenNodes, int paramNumberOutputNodes){
  individualstruct->numberOfHiddenNodes = paramNumberHiddenNodes;
  individualstruct->numberOfOutputNodes = paramNumberOutputNodes; 
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

//This function simply prints the values currently contained in the fields of a Node struct
void printNode(Node* paramNode){
	printf("Node contains input value %f, weight: %f, bias: %f, output: %f, lms: %f \n",
	paramNode->input, paramNode->weight,paramNode->bias, paramNode->output, paramNode->lms);
}

void printFFANN(Individual* citizen){

		//first print input node details
		printf("Input layer contains:\n");
		printNode(citizen->inputLayer);
		
		
		//now print hidden layer node details
		printf("Hidden layer contains %d nodes:\n",citizen->numberOfHiddenNodes);
		int c = 0; 
		for(c=0;c<citizen->numberOfHiddenNodes; c++){
				printNode( (Node*) citizen->hiddenLayer[c] );
		}
		
		//now print output layer node details
		printf("Output layer contains:\n");
		for(c=0;c<citizen->numberOfOutputNodes; c++){
				printNode( (Node*) citizen->outputLayer[c] );
		}
}


void constructFFANN(Population* populationStruct, int memberNumber){
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
 
#ifdef TEST
printf("Just about to iterate over all hidden layer nodes and create each one...\n");
#endif 

//Unfortunately this cannot be done in a loop because C needs all variable names at compile time :(
	Node hiddenNode0;
	Node hiddenNode1;
	Node hiddenNode2;
	Node hiddenNode3;
	Node hiddenNode4;
	Node hiddenNode5;
	Node hiddenNode6;
	Node hiddenNode7;
	Node hiddenNode8;
	Node hiddenNode9;
	Node hiddenNode10;
	Node hiddenNode11;
	Node hiddenNode12;
	Node hiddenNode13;
	Node hiddenNode14;
	Node hiddenNode15;
	Node hiddenNode16;
	Node hiddenNode17;
	Node hiddenNode18;
	Node hiddenNode19;
	
	constructNode(&hiddenNode0);
  hiddenLayer[0] = (Node*) &hiddenNode0;
  
  constructNode(&hiddenNode1);
  hiddenLayer[1] = (Node*) &hiddenNode1;
  
  constructNode(&hiddenNode2);
  hiddenLayer[2] = (Node*) &hiddenNode2;
  
  constructNode(&hiddenNode3);
  hiddenLayer[3] = (Node*) &hiddenNode3;
  
  constructNode(&hiddenNode4);
  hiddenLayer[4] = (Node*) &hiddenNode4;
  
  constructNode(&hiddenNode5);
  hiddenLayer[5] = (Node*) &hiddenNode5;
  
  constructNode(&hiddenNode6);
  hiddenLayer[6] = (Node*) &hiddenNode6;
  
  constructNode(&hiddenNode7);
  hiddenLayer[7] = (Node*) &hiddenNode7;
  
  constructNode(&hiddenNode8);
  hiddenLayer[8] = (Node*) &hiddenNode8;
  
  constructNode(&hiddenNode9);
  hiddenLayer[9] = (Node*) &hiddenNode9;
  
  constructNode(&hiddenNode10);
  hiddenLayer[10] = (Node*) &hiddenNode10;
  
  constructNode(&hiddenNode11);
  hiddenLayer[11] = (Node*) &hiddenNode11;
  
  constructNode(&hiddenNode12);
  hiddenLayer[12] = (Node*) &hiddenNode12;
  
  constructNode(&hiddenNode13);
  hiddenLayer[13] = (Node*) &hiddenNode13;
  
  constructNode(&hiddenNode14);
  hiddenLayer[14] = (Node*) &hiddenNode14;
  
  constructNode(&hiddenNode15);
  hiddenLayer[15] = (Node*) &hiddenNode15;
  
  constructNode(&hiddenNode16);
  hiddenLayer[16] = (Node*) &hiddenNode16;
  
  constructNode(&hiddenNode17);
  hiddenLayer[17] = (Node*) &hiddenNode17;
  
  constructNode(&hiddenNode18);
  hiddenLayer[18] = (Node*) &hiddenNode18;
  
  constructNode(&hiddenNode19);
  hiddenLayer[19] = (Node*) &hiddenNode19;
	
 /*for(c=0; c < numberOfHidden; c++){ //#x in range(numberOfHidden):
   Node newnode; // = malloc(nodeSizeMemory);
   constructNode(&newnode);
   hiddenLayer[c] = (Node*) &newnode;
 }*/
  printf("length of hidden layer inside constructFFANN : %d \n",numberOfHidden);
#ifdef TEST
printf("Just about to go move onto the output layer...\n");
#endif

 Node** outputLayer; // #3 nodes, one per expected output
 outputLayer = (Node**) malloc( nodeSizeMemory * defaultNumberOutputNodes );
 
	Node outputNode0;
	Node outputNode1;
	Node outputNode2;
	constructNode( &outputNode0 );
	outputLayer[0] = (Node*) &outputNode0;
	constructNode( &outputNode1 );
	outputLayer[1] = &outputNode1;
	constructNode( &outputNode2 );
	outputLayer[2] = &outputNode2;
	
#ifdef TEST
printf("Just about to iterate throgh output layer...\n");
#endif
 
 //#now create output node weights, the same amount as there are hidden nodes
 for(c=0; c < defaultNumberOutputNodes; c++){
   int d = 0;
   for(d=0; d < numberOfHidden; d++){ 

		 outputLayer[c]->weights[d] = 0.0f; //#clear the weights
		 //#need same number of weights in each output node as there are hidden nodes...

			 outputLayer[c]->weights[d] = getRandomWeightValueFloat(); //#initialise weights randomly
			}
		}
		
#ifdef TEST
printf("Just finished constructing output layer...\n");
#endif


#ifdef TEST
printf("Just about to add individual to population...\n");
#endif
 //#Now add to the current population list
 Individual citizen;
 constructIndividual(&citizen,numberOfHidden,defaultNumberOutputNodes);
 citizen.inputLayer = &inputnode;
 citizen.hiddenLayer = hiddenLayer;
 citizen.outputLayer = outputLayer;
 
#ifdef TEST
printf("Just put the parts of the individual together...\n");
#endif 

 populationStruct->oldpopulation[memberNumber] = (Individual*) &citizen;

#ifdef TEST
printf("Just added the individual to the oldpopulation...\n");
#endif 
 
 
 
#ifdef TEST
printf("Just about to print out that citizen's details...\n");
#endif 

	//printFFANN(populationStruct->oldpopulation[memberNumber]);
	
 /*for(c=0;c<popsize;c++){
	Individual citizen;
	constructIndividual(&citizen);
	citizen.inputLayer = &inputnode;
	citizen.hiddenLayer = hiddenLayer;
	citizen.outputLayer = outputLayer;
	populationStruct->oldpopulation[c] = (Individual*) &citizen;
 }*/
 
}


