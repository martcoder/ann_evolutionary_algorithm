#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef MATH_H
#define MATH_H 1
#include<math.h>
#endif

#include "individual_aux.c"

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


//Activation functions: https://www.geeksforgeeks.org/activation-functions-neural-networks/



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
	//constructIndividual( superpopulation.oldpopulation[memberNumber] );
 
 constructNode(populationStruct->oldpopulation[memberNumber]->inputLayer);
 //# inputLayer = Node() 
 int numberOfHidden = getRandomNumberHiddenNodesInt();
 //#print("number of hidden: "+str(numberOfHidden))

#ifdef TEST
printf("Next to construct hidden layer....\n");
#endif

 //#construct hidden layer
 //Node** hiddenLayer;
 //hiddenLayer = (Node**) malloc( nodeSizeMemory * numberOfHidden );
 //int c = 0;
 
#ifdef TEST
printf("Just about to iterate over all hidden layer nodes and create each one...\n");
#endif 

//Unfortunately this cannot be done in a loop because C needs all variable names at compile time :(
	
	int h = 0;
	for(h=0; h < numberOfHidden; h++){
	  constructNode(populationStruct->oldpopulation[memberNumber]->hiddenLayer[h]);
   }	
  
 /*for(c=0; c < numberOfHidden; c++){ //#x in range(numberOfHidden):
   Node newnode; // = malloc(nodeSizeMemory);
   constructNode(&newnode);
   hiddenLayer[c] = (Node*) &newnode;
 }*/
  printf("length of hidden layer inside constructFFANN : %d \n",numberOfHidden);
#ifdef TEST
printf("Just about to go move onto the output layer...\n");
#endif

 //Node** outputLayer; // #3 nodes, one per expected output
 //outputLayer = (Node**) malloc( nodeSizeMemory * defaultNumberOutputNodes );
 
	int o = 0;
	for(o=0; o < outputLayerLength; o++){
#ifdef TEST
printf("Just about to do CONSTRUCTNODE on an output layer node...\n");
#endif
	  constructNode(populationStruct->oldpopulation[memberNumber]->outputLayer[o]);
   }	
#ifdef TEST
printf("Just about to iterate throgh output layer...\n");
#endif
 
 //#now create output node weights, the same amount as there are hidden nodes
 int c = 0;
 int d = 0;
 for(c=0; c < defaultNumberOutputNodes; c++){
   
   for(d=0; d < numberOfHidden; d++){ 

		 //outputLayer[c]->weights[d] = 0.0f; //#clear the weights
		 //#need same number of weights in each output node as there are hidden nodes...

			 populationStruct->oldpopulation[memberNumber]->outputLayer[c]->weights[d] = getRandomWeightValueFloat(); //#initialise weights randomly
			}
		}
		
#ifdef TEST
printf("Just finished constructing output layer...\n");
#endif


#ifdef TEST
printf("Just about to add individual to population...\n");
#endif
 //#Now add to the current population list
 //Individual citizen;
 constructIndividual(populationStruct->oldpopulation[memberNumber],numberOfHidden,defaultNumberOutputNodes);
 /*populationStruct->oldpopulation[memberNumber]->inputLayer = &inputnode;
 populationStruct->oldpopulation[memberNumber]->hiddenLayer = hiddenLayer;
 populationStruct->oldpopulation[memberNumber]->outputLayer = outputLayer;*/
 
#ifdef TEST
printf("Just put the parts of the individual together...\n");
#endif 

 

#ifdef TEST
printf("Just added the individual to the oldpopulation...\n");
#endif 
 
 
/* 
#ifdef TEST
printf("Just about to PRINT OUT THAT CITIZEN's DETAILS...\n");
#endif 

	printFFANN(populationStruct->oldpopulation[memberNumber]);
	*/
 /*for(c=0;c<popsize;c++){
	Individual citizen;
	constructIndividual(&citizen);
	citizen.inputLayer = &inputnode;
	citizen.hiddenLayer = hiddenLayer;
	citizen.outputLayer = outputLayer;
	populationStruct->oldpopulation[c] = (Individual*) &citizen;
 }*/
 
}


