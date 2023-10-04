#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <stdbool.h>

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
  //Ensure that no division by zero happens....
  if(a == 0.0){ a = 0.0000001f;}
  if(b == 0.0){ b = 0.0000001f;}
  if(c == 0.0){ c = 0.0000001f;}
  
  //Now calculate LMS depending on which value is the largest...
  // e.g. for LOW pressure the a value should ideally be largest
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
        
        while(!fieldEndFound){ // look through line of chars for the comma delimiter
					if( lineOfData[fieldIndex] == ',')
						fieldEndFound = 1;
					else{
					  value[fieldIndex] = lineOfData[fieldIndex];
						fieldIndex++; 
					}
				}
				fieldIndex++;
				value[fieldIndex] = '\0'; // put the foudn value into this array
				(*result) = (float) atof(value); // converter found value into a float
				
				if(  ( (float) floatAbs(*result) ) >  normaliseCeiling ){ // for data over upper ceiling
						(*result) = 1.0f; // return max absolute normalised value
				}
				else{ // for data under the upper ceiling, get absolute, normalise it and return
					(*result) =  ( (float) floatAbs(*result) ) / normaliseCeiling;
				}
}

//So gonna pass in each data file one by one, so the iteration over different data will happen in main
void process( char * filename, int listLength, int member, float expectedResultLow, float expectedResultMed, float expectedResultHigh, float normaliseCeiling  ){
	int c = 0; 
	
#ifdef TEST
printf("Just started process function...\n");
#endif
	
		FILE *fp;
    char *line = NULL;
    size_t len = 0;
    ssize_t read;

   fp = fopen(filename, "r");
    if (fp == NULL){
				printf("An issue occured when reading in the data file\n");
        exit(EXIT_FAILURE);
		}
		
		
#ifdef TEST
printf("Just about to read datafile  line by line...\n");
#endif
	 
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
     // Refer: https://stackoverflow.com/questions/3501338/c-read-file-line-by-line
		fclose(fp);
		if( line ){
			free(line);
		}
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

void writeFFANNtoFile(Individual* citizen){
		FILE* fp;
		fp = fopen("log.txt", "a");
		fprintf(fp, "========Best ANN Found Details =========: \n");
		fprintf(fp, "Input layer: \n");
		fprintf(fp, "Input weight: %f, input bias: %f \n",citizen->inputLayer->weight, citizen->inputLayer->bias);
		fprintf(fp, "Input weight: %f, input bias: %f \n",citizen->inputLayer->weight, citizen->inputLayer->bias);
		int c,d; 
		for(c=0; c < citizen->numberOfHiddenNodes; c++){
			fprintf(fp, "\nHidden layer node %d:- weight: %f, bias: %f\n",c, citizen->hiddenLayer[c]->weight, citizen->hiddenLayer[c]->bias);
			
		}
		
		for(c=0; c < citizen->numberOfOutputNodes; c++){
				fprintf(fp, "\nOutput layer node %d :-  bias: %f \n", c, citizen->outputLayer[c]->bias);
				fprintf(fp, "Weights: \n");
				for(d=0; d < citizen->numberOfHiddenNodes; d++){
					fprintf( fp, "w%d=%f, ",d,citizen->outputLayer[c]->weights[d]);
				}
				fprintf(fp, "\n");
		}
		
		fprintf(fp,"\nLMS is: %f\n\n", citizen->lms);

    // close file
    fclose(fp);
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
		printf("++++++ ===Individual's LMS is currently: %f ===+++++++\n\n",citizen->lms);
}

void constructFFANN(Individual** populationStruct, int memberNumber){
	//# construct input layer, aka a single Node 
#ifdef TEST
printf("Just about to go through constructFFANN function, first to construct input node...\n");
#endif
	//constructIndividual( superpopulation.oldpopulation[memberNumber] );
 
 constructNode(populationStruct[memberNumber]->inputLayer);
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
	  constructNode(populationStruct[memberNumber]->hiddenLayer[h]);
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
	  constructNode(populationStruct[memberNumber]->outputLayer[o]);
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

			 populationStruct[memberNumber]->outputLayer[c]->weights[d] = getRandomWeightValueFloat(); //#initialise weights randomly
			}
		}
		
#ifdef TEST
printf("Just finished constructing output layer...\n");
#endif



 //#Now add to the current population list
 //Individual citizen;
 constructIndividual(populationStruct[memberNumber],numberOfHidden,defaultNumberOutputNodes);
 /*populationStruct->oldpopulation[memberNumber]->inputLayer = &inputnode;
 populationStruct->oldpopulation[memberNumber]->hiddenLayer = hiddenLayer;
 populationStruct->oldpopulation[memberNumber]->outputLayer = outputLayer;*/
 
#ifdef TEST
printf("Just put the parts of the individual together...\n");
#endif 

}

void bubbleSort(Individual** arr, int n) //inspired by https://www.geeksforgeeks.org/bubble-sort/ 
{
	
#ifdef AUXTEST
	printf("Inside bubbleSort function\n");
#endif

    int i, j;
    bool swapped;
    for (i = 0; i < n - 1; i++) {
        swapped = false;
        for (j = 0; j < n - i - 1; j++) {
#ifdef AUXTEST					
					printf("i is %d and j is %d\n",i, j);
#endif
            if (arr[j]->lms > arr[j + 1]->lms) {
                copyIndividual( arr[j + 1], &indSort);
								copyIndividual( arr[j], arr[j+1] );
								copyIndividual( &indSort, arr[j] ); 
                swapped = true;
            }
        }
 
        // If no two elements were swapped
        // by inner loop, then break
        if (swapped == false)
            break;
    }
#ifdef AUXTEST
		printf("Finished bubbleSort function\n");
#endif
}


void tournament(Population* superpopulation, int newpopMemberIndex){
	printf("Just about to do a tournament selection\n");
	//select individuals for tournament ... assuming tournamentSize of 4
	int index1, index2, index3, index4; 
	index1 = getRandomIndividualIndex();
	index2 = getRandomIndividualIndex();
	while(index2 == index1){
		index2 = getRandomIndividualIndex();
	}
	index3 = getRandomIndividualIndex();
	while( (index3 == index2) || (index3 == index1) ){
		index3 = getRandomIndividualIndex();
	}
	index4 = getRandomIndividualIndex(); 
	while( (index4 == index3) || (index4 == index2) || (index4 == index1) ){
		index4 = getRandomIndividualIndex();
	}
	
	Individual** tournArray = (Individual**) malloc( individualSizeMemory * tournamentSize );
	
	#ifdef AUXTEST
	printf("Chosen tournament member indexes are %d %d %d %d\n",index1, index2, index3, index4);
	#endif
	
	#ifdef AUXTEST
	printf("About to copy indiv at index %d into tournArray[0]\n",index1);
	#endif
	
	tournArray[0] = &indTourn0;
	copyIndividual(superpopulation->oldpopulation[index1], tournArray[0]);
	
	#ifdef AUXTEST
	printf("About to copy indiv at index %d into tournArray[1]\n",index2);
	#endif
	
	tournArray[1] = &indTourn1;
	copyIndividual(superpopulation->oldpopulation[index2], tournArray[1]);
	
	#ifdef AUXTEST
	printf("About to copy indiv at index %d into tournArray[2]\n",index3);
	#endif
	
	tournArray[2] = &indTourn2;
	copyIndividual(superpopulation->oldpopulation[index3], tournArray[2] );
	
	#ifdef AUXTEST
	printf("About to copy indiv at index %d into tournArray[3]\n",index4);
	#endif
	
	tournArray[3] = &indTourn3;
	copyIndividual( superpopulation->oldpopulation[index4], tournArray[3]);
	
	bubbleSort(tournArray, 4); 

#ifdef AUXTEST
	printf("sorted tournarray is: \n");
	printFFANN(tournArray[0]);
	printFFANN(tournArray[1]);
	printFFANN(tournArray[2]);
	printFFANN(tournArray[3]);
#endif
	
	// Now do breeding with probability, e.g. just take one of the best 2 or with prob do breeding between best 2 of tournament
	float prob = getRandomBiasValueFloat(2.5f, -7.5f); // will get a positive or negative value
	if(prob > 0.0f){ //25% chance of just copying individual as is... very boring thing to happen
		copyIndividual(tournArray[0], superpopulation->newpopulation[newpopMemberIndex]); // just copy a parent to new generation
	}
	else{ // Breed, by copying input and output layer from one parent, and hidden layer from other parent!
		// Firstly copy everything from parent0
		copyIndividual(tournArray[0], superpopulation->newpopulation[newpopMemberIndex]);
		
		// Now overwrite the hidden layer with details from parent1
		superpopulation->newpopulation[newpopMemberIndex]->numberOfHiddenNodes = tournArray[1]->numberOfHiddenNodes;
		int n = 0; 
		for(n=0; n < tournArray[1]->numberOfHiddenNodes; n++){
			superpopulation->newpopulation[newpopMemberIndex]->hiddenLayer[n]->weight = tournArray[1]->hiddenLayer[n]->weight;
			superpopulation->newpopulation[newpopMemberIndex]->hiddenLayer[n]->bias = tournArray[1]->hiddenLayer[n]->bias;
		}
		
		//Now update output layer weights array for each output node to be same length as hidden layer
		int o = 0;
		for(o=0; o < superpopulation->newpopulation[newpopMemberIndex]->numberOfOutputNodes; o++){
			for(n=0; n < superpopulation->newpopulation[newpopMemberIndex]->numberOfHiddenNodes; n++){
				#ifdef AUXTEST
					printf("weight in outputNode %d is currently %f\n",o,superpopulation->newpopulation[newpopMemberIndex]->outputLayer[o]->weights[n]);
				#endif
					if( (-0.0001f < superpopulation->newpopulation[newpopMemberIndex]->outputLayer[o]->weights[n]) 
							&& 
							( superpopulation->newpopulation[newpopMemberIndex]->outputLayer[o]->weights[n] < 0.0001f )
						){ // Checking if the weight value is zero, then will need to set a value. e.g. if parent0 hiddenLayer was shorter than parent1 hidden layer
							superpopulation->newpopulation[newpopMemberIndex]->outputLayer[o]->weights[n] = getRandomWeightValueFloat();
					}
					#ifdef AUXTEST
					printf("If setting was necessary that same weight in outputNode %d is currently %f\n",o,superpopulation->newpopulation[newpopMemberIndex]->outputLayer[o]->weights[n]);
					#endif
			}
		}
	}
}

void mutate(Individual* member){
		printf("////****////  Mutating... ////****/////***** \n");
			float prob = getRandomBiasValueFloat(3.0f, -3.0f);
			if( prob < 0.0f ){ //50% chance of mutation
					member->inputLayer->weight = getRandomWeightValueFloat();
			}
			
			prob = getRandomBiasValueFloat(3.0f, -3.0f);
			if( prob < 0.0f ){ //50% chance of mutation
					member->inputLayer->bias = getRandomWeightValueFloat();
			}
			
			int h = 0;
			for(h = 0; h < member->numberOfHiddenNodes; h++){
					prob = getRandomBiasValueFloat(3.0f, -3.0f);
					if( prob < 0.0f ){ //50% chance of mutation
							member->hiddenLayer[h]->weight = getRandomWeightValueFloat();
					}
					prob = getRandomBiasValueFloat(3.0f, -3.0f);
					if( prob < 0.0f ){ //50% chance of mutation
							member->hiddenLayer[h]->bias = getRandomWeightValueFloat();
					}
			}
			
			int o = 0;
			for(o=0; o < member->numberOfOutputNodes; o++){
					prob = getRandomBiasValueFloat(3.0f, -3.0f);
					if( prob < 0.0f ){ //50% chance of mutation
								member->outputLayer[o]->bias = getRandomWeightValueFloat();
					}
					for(h=0; h < member->numberOfHiddenNodes; h++){
						prob = getRandomBiasValueFloat(3.0f, -3.0f);
						if( prob < 0.0f ){ //50% chance of mutation
								member->outputLayer[o]->weights[h] = getRandomWeightValueFloat();
						}
					}
			}
}
