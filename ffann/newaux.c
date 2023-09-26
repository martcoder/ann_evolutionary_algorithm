#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef MATH_H
#define MATH_H 1
#include<math.h>
#endif

#include "individual_aux.c"

//#define TEST 1

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


float normalisedLms( float a, float b, float c, float expectedA, float expectedB, float expectedC){
  float x,y,z;
  float lmsA, lmsB, lmsC;
	if( (a > b) && (a > c) ){ // a largest
		x = 1.0f; // a / a
		y = a/b;
		z = a/c;
		
		lmsA = floatAbs(x - expectedA);
		lmsB = floatAbs(y - expectedB);
		lmsC = floatAbs(z - expectedC);
		
		return (lmsA + lmsB + lmsC) * 0.33333f;
		
	}
	
	if( (b > a) && (b > c) ){ // b largest
		x = b / a;
		y = 1.0f; // b / b
		z = b / c;
		
		lmsA = floatAbs(x - expectedA);
		lmsB = floatAbs(y - expectedB);
		lmsC = floatAbs(z - expectedC);
		
		return (lmsA + lmsB + lmsC) * 0.33333f;
	}
	
	if( (c > b) && (c > a) ){ // c largest
		x = c / a;
		y = c / b;
		z = 1.0f; // c/c;
		
		lmsA = floatAbs(x - expectedA);
		lmsB = floatAbs(y - expectedB);
		lmsC = floatAbs(z - expectedC);
		
		return (lmsA + lmsB + lmsC) * 0.33333f;
	}
	
}

//This function takes the line of data as a string, and extracts the first value as a float
// it also normalises the data
void getFirstFloat(char * lineOfData, float * result, float normaliseCeiling){
	
				char * startOfField;
        char value[16];
				int fieldEndFound = 0;
				int fieldIndex = 0;
        
        while(!fieldEndFound){
					if( lineOfData[fieldIndex] == ',')
						fieldEndFound = 1;
					else{
					  value[fieldIndex] = lineOfData[fieldIndex];
						fieldIndex++; 
					}
				}
				fieldIndex++;
				value[fieldIndex] = '\0';
				(*result) = (float) atof(value);
				
				if(  ( (float) floatAbs(*result) ) >  normaliseCeiling ){
						(*result) = 1.0f;
				}
				else{ 
					(*result) =  ( (float) floatAbs(*result) ) / normaliseCeiling;
				}
}

//So gonna pass in each data file one by one, so the iteration over different data will happen in main
void process( char * filename, int listLength, int member, float expectedResultLow, float expectedResultMed, float expectedResultHigh, float normaliseCeiling  ){
	int c = 0; 
	
		FILE *fp;
    char *line = NULL;
    size_t len = 0;
    ssize_t read;

   fp = fopen(filename, "r");
    if (fp == NULL){
				printf("An issue occured when reading in the data file\n");
        exit(EXIT_FAILURE);
		}
	 
	 float dataCount = 0;
   while ((read = getline(&line, &len, fp)) != -1) {
				dataCount++;
        //printf("Retrieved line of length %zu :\n", read);
        //printf("%s", line);
        float float_data;
        getFirstFloat(line, &float_data, normaliseCeiling); // converts the first data value from the string line into a float
        //printf("Float data value is %f\n", float_data);
				
				//=====INPUT NODE PROCESSING FIRST=========
				// Decorate the input with the data value
				superpopulation.oldpopulation[member]->inputLayer->input = float_data;
				
				//Multiply input by weight
				superpopulation.oldpopulation[member]->inputLayer->output = superpopulation.oldpopulation[member]->inputLayer->input * superpopulation.oldpopulation[member]->inputLayer->weight;
				
				//Add the bias
				superpopulation.oldpopulation[member]->inputLayer->output += superpopulation.oldpopulation[member]->inputLayer->bias;
				
				//Run through the activation function
				superpopulation.oldpopulation[member]->inputLayer->output = relu(superpopulation.oldpopulation[member]->inputLayer->output);
				
				//==========HIDDEN LAYER PROCESSING============
				//Now do the same for each hidden-layer node.....
				int h = 0;
				for(h = 0; h < superpopulation.oldpopulation[member]->numberOfHiddenNodes; h++){
					//decorate hidden node input from input node's output
					superpopulation.oldpopulation[member]->hiddenLayer[h]->input = superpopulation.oldpopulation[member]->inputLayer->output;
					// Multiply input by weight
					superpopulation.oldpopulation[member]->hiddenLayer[h]->output = superpopulation.oldpopulation[member]->hiddenLayer[h]->input * superpopulation.oldpopulation[member]->hiddenLayer[h]->weight;
					// Add in bias
					superpopulation.oldpopulation[member]->hiddenLayer[h]->output += superpopulation.oldpopulation[member]->hiddenLayer[h]->bias;
					
					//Run through activation function
					superpopulation.oldpopulation[member]->hiddenLayer[h]->output = relu(superpopulation.oldpopulation[member]->hiddenLayer[h]->output);
				}

				//=======OUTPUT LAYER PROCESSING============
				int o, w;
				for(o = 0; o < superpopulation.oldpopulation[member]->numberOfOutputNodes; o++){
						superpopulation.oldpopulation[member]->outputLayer[o]->output = 0.0f; // clear output value 
						// Sum each result of multiplying input from preceding layer with appropriate weight
						for(w = 0; w < superpopulation.oldpopulation[member]->numberOfHiddenNodes; w++){
								// input value (which is output of preceding layer node) * appropriate weight....
								superpopulation.oldpopulation[member]->outputLayer[o]->output += superpopulation.oldpopulation[member]->hiddenLayer[w]->output * superpopulation.oldpopulation[member]->outputLayer[o]->weights[w];
						}
						
						// Now add in bias
						superpopulation.oldpopulation[member]->outputLayer[o]->output += superpopulation.oldpopulation[member]->outputLayer[o]->bias;
					
						// Run through activation function
						superpopulation.oldpopulation[member]->outputLayer[o]->output = relu(superpopulation.oldpopulation[member]->outputLayer[o]->output);
				}
				
				//=============FIND NORMALISED LMS of ANN AFTER PROCESSING THAT LINE OF DATA=====================
				superpopulation.oldpopulation[member]->lms += normalisedLms( superpopulation.oldpopulation[member]->outputLayer[0]->output, superpopulation.oldpopulation[member]->outputLayer[1]->output, superpopulation.oldpopulation[member]->outputLayer[2]->output, expectedResultLow, expectedResultMed, expectedResultHigh) / dataCount;
				//printf("LMS is current %f\n",superpopulation.oldpopulation[member]->lms);
    }

   free(line);
   //exit(1); // for exiting in order to see result so far without iterating through all the data!!!
		
}

//This function simply prints the values currently contained in the fields of a Node struct
void printNode(Node* paramNode, int printWeightsArray, int numberHidden){
	printf("Node contains input value %f, weight: %f, bias: %f, output: %f, lms: %f \n",
	paramNode->input, paramNode->weight,paramNode->bias, paramNode->output, paramNode->lms);
	if(printWeightsArray){
		int w = 0;
		printf("Output node contains weights array of: ");
		for(w=0; w < 	numberHidden; w++){
				printf("%f, ",paramNode->weights[w]);
		}
		printf("\n");
	}
}

void printFFANN(Individual* citizen){

		//first print input node details
		printf("Input layer contains:\n");
		printNode(citizen->inputLayer, 0, 0);
		
		
		//now print hidden layer node details
		printf("Hidden layer contains %d nodes:\n",citizen->numberOfHiddenNodes);
		int c = 0; 
		for(c=0;c<citizen->numberOfHiddenNodes; c++){
				printNode( (Node*) citizen->hiddenLayer[c], 0, 0 );
		}
		
		//now print output layer node details
		printf("Output layer contains:\n");
		int w = 0;
		for(c=0;c<citizen->numberOfOutputNodes; c++){
				printNode( (Node*) citizen->outputLayer[c],1, citizen->numberOfHiddenNodes );
		}
		
		//finally print lms detail
		printf("Individual's LMS is currently: %f\n",citizen->lms);
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


