#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <stdbool.h>

#ifndef MATH_H
#define MATH_H 1
#include<math.h>
#endif

#include "declarations_aux.c"

//#define TEST 1
//#define DEEPTEST 1

//https://medium.com/@b.terryjack/introduction-to-deep-learning-feed-forward-neural-networks-ffnns-a-k-a-c688d83a309d 
//https://en.wikipedia.org/wiki/Tournament_selection
//https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)
//Activation functions: https://www.geeksforgeeks.org/activation-functions-neural-networks/

void initialiseVariables(){
	
	linR = 0; // Default is that linear regression is not being solved, in main sets to 1 if linear regression cmd line arg set
	oneTime = 0; // Default of 0, generally not running data through an ANN one-time, but processing for the algorithm
	expectedResultRegression = 0.0f;
	numberActivationFunctions = 3;
	
	numberOfLowDataFiles = 3;
	numberOfMedDataFiles = 4;
	numberOfHighDataFiles = 3;
	defaultNumberOutputNodes = 3;
	bestlms = 1000000000000000000.0; // assigning initial high value
	popsize = 99;
	hiddenMax = 20;
	hiddenMin = 5;
	outputLayerLength = 3;
	weightMax = 5.0f;
	//Set global variables values
	lmsResult = (float*) malloc(sizeof(float) * popsize);
	numCycles = 2; //global variable
	nodeSizeMemory = ( sizeof(float) * 6 ) + (sizeof(int)) + ( sizeof(float) * hiddenMax );
	individualSizeMemory = popsize * ( nodeSizeMemory + (hiddenMax * nodeSizeMemory) + (nodeSizeMemory * outputLayerLength) + (sizeof(float) * 5) );
	
	elitism = 1 + ( popsize / 10.0 );
	tournamentSize = 8;
}

float floatAbs(float value){
	if( value < 0.0f ){
			return value * -1.0f;
	}
	else{
			return value;
	}
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
	if(value <= 0.0f){
			return 0.0f;
	}
	else{
		return value; // #accel data has plenty of negative values, so using absolute
	}
}

float activation_cosh(float value){
		return cosh(value);
}

int getRandomNumberHiddenNodesInt(){
	return (rand() % (hiddenMax - hiddenMin + 1)) + hiddenMin;
}

int getRandomIndividualIndex(){
		return (rand() % (popsize));
}

int getRandomActivationFunction(){
		return (rand() % (numberActivationFunctions - 1 + 1)) + 1;
}

float processActivationFunction(Node* nodeactivate, float input){
		if( nodeactivate->activationFunction == 1){
				return sigmoid(input);
		}
		if( nodeactivate->activationFunction == 2){
				return relu(input);
		}
		if( nodeactivate->activationFunction == 3){
				return activation_cosh(input);
		}
}

void constructNode(Node * nodestruct){
	nodestruct->input = 0.0f;
	nodestruct->weight = getRandomWeightValueFloat();
	nodestruct->bias = getRandomBiasValueFloat(weightMax/5.0, weightMax/-5.0);
	nodestruct->weights = (float *) malloc( sizeof(float) * hiddenMax );
	nodestruct->output = 0.0f;
	nodestruct->lms = 2.0f; // initialising to a high value...
	nodestruct->activationFunction = getRandomActivationFunction(); 
}

void constructIndividual(Individual * individualstruct, int paramNumberHiddenNodes, int paramNumberOutputNodes){
  individualstruct->numberOfHiddenNodes = paramNumberHiddenNodes;
  individualstruct->numberOfOutputNodes = paramNumberOutputNodes; 
	//individualstruct->inputLayer = paramInputlayer; //(Node *) malloc(nodeSizeMemory);
	//individualstruct->hiddenLayer = paramHiddenlayer; //(Node **) malloc(nodeSizeMemory * individualstruct->numberOfHiddenNodes);
	//individualstruct->outputLayer = paramOutputlayer; //(Node **) malloc(nodeSizeMemory * individualstruct->numberOfOutputNodes);
	individualstruct->best = 0.0f;
	individualstruct->lms = 4.0f;
	individualstruct->lmsLOW = 4.0f;
	individualstruct->lmsMED = 4.0f;
	individualstruct->lmsHIGH = 4.0f;
}

void constructPopulation(Population * populationstruct){
  populationstruct->oldpopulation = (Individual**) malloc( individualSizeMemory * popsize );
	populationstruct->newpopulation = (Individual**) malloc( individualSizeMemory * popsize );
	populationstruct->miscpopulation = (Individual**) malloc( individualSizeMemory * (tournamentSize + 2) );
}

void copyIndividual(Individual* from, Individual* to){
#ifdef AUXTEST
	printf("Inside copy function\n");
#endif
		//====FIRSTLY OVERALL INDIVIDUAL VALUES
		
		(*to).lms = (*from).lms;
		
#ifdef AUXTEST
		printf("Just set lms inside copy function\n");
#endif
		to->numberOfHiddenNodes = from->numberOfHiddenNodes;
		to->numberOfOutputNodes = from->numberOfOutputNodes;
#ifdef AUXTEST
		printf("about to copy input layer\n");
#endif
		//=====FIRSTLY THE INPUT LAYER

		to->inputLayer->input = 0.0f;
		to->inputLayer->output = 0.0f;
		to->inputLayer->weight = from->inputLayer->weight;
		to->inputLayer->bias = from->inputLayer->bias;
		to->inputLayer->activationFunction = from->inputLayer->activationFunction;
		
#ifdef AUXTEST
		printf("about to copy hidden layer\n");
#endif

		//======NOW THE HIDDEN LAYER===========
		int c = 0; 
		for(c=0; c < from->numberOfHiddenNodes; c++){
				to->hiddenLayer[c]->input = 0.0f;
				to->hiddenLayer[c]->output = 0.0f;
				to->hiddenLayer[c]->weight = from->hiddenLayer[c]->weight;
				to->hiddenLayer[c]->bias = from->hiddenLayer[c]->bias;
				to->hiddenLayer[c]->activationFunction = from->hiddenLayer[c]->activationFunction;
		}
		
#ifdef AUXTEST
		printf("about to copy output layer\n");
#endif

		//======FINALLY THE OUTPUT LAYER=======
		int w = 0;
		for(c=0; c < from->numberOfOutputNodes; c++){
				to->outputLayer[c]->input = 0.0f;
				to->outputLayer[c]->output = 0.0f;
				to->outputLayer[c]->weight = from->outputLayer[c]->weight;
				to->outputLayer[c]->bias = from->outputLayer[c]->bias;
				to->outputLayer[c]->activationFunction = from->outputLayer[c]->activationFunction;
				for(w=0;w< from->numberOfHiddenNodes; w++){
						to->outputLayer[c]->weights[w] = from->outputLayer[c]->weights[w];
				}
		}
#ifdef AUXTEST
		printf("Copying complete\n");
#endif
}

float normalisedLms_linearRegression( float a, float expectedA ){
		float lmsA;
		if(a == 0.0){ a = 0.0000001f;}
		lmsA = (a - expectedA) * (a - expectedA);
#ifdef DEEPTEST
		printf("expected is %f and actual is %f\n",expectedA, a);
#endif
		return lmsA;
}

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
void getFirstFloat(char * lineOfData, float * result, float normaliseCeiling, int linearReg){
	
				char * startOfField;
        char value[16];
				int fieldEndFound = 0;
				int fieldIndex = 0;
        
        while(!fieldEndFound){ // look through line of chars for the comma delimiter
					if( lineOfData[fieldIndex] == ','){
						fieldEndFound = 1;
					}
					else{
					  value[fieldIndex] = lineOfData[fieldIndex];
						fieldIndex++; 
					}
				}
				fieldIndex++;
				value[fieldIndex] = '\0'; // put the foudn value into this array
				
				(*result) = (float) atof(value); // converter found value into a float
				
				if(linearReg != 1){
					if(  ( (float) floatAbs(*result) ) >  normaliseCeiling ){ // for data over upper ceiling
								(*result) = 1.0f; // return max absolute normalised value
					}
					else{ // for data under the upper ceiling, get absolute, normalise it and return
							(*result) =  ( (float) floatAbs(*result) ) / normaliseCeiling;
					}
				}
}

//So gonna pass in each data file one by one, so the iteration over different data will happen in main
void process( char * filename, char * filenameWrite, int member, int linearRegression, float expectedResultLow, float expectedResultMed, float expectedResultHigh, float normaliseCeiling, int oneTime  ){
	int c = 0; 
	
#ifdef TEST
printf("Just started process function...\n");
#endif
	
		FILE *fp; // This is for reading from data file
		FILE *fpW; // This is for writing the ANN outputs to logfile
		
		
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    
		fp = fopen(filename, "r");
    if (fp == NULL){
				printf("An issue occured when reading in the data file\n");
        exit(EXIT_FAILURE);
		}
		
		if(oneTime == 1){ // e.g. one off processing of data throguh ANN, e.g. to log output from best ANN once algorithm done
			fpW = fopen(filenameWrite, "a");
			if( fpW == NULL ){
					printf("An issue occurred when opening file to write\n");
					exit(EXIT_FAILURE);
			}
		}
		
		
#ifdef TEST
printf("Just about to read datafile  line by line...\n");
#endif
	 
	 float dataCount = 0;
   while ((read = getline(&line, &len, fp)) != -1) {
				dataCount++;
				
#ifdef DEEPTEST
        printf("Retrieved line of length %zu :\n", read);
        printf("%s", line);
#endif
        float float_data;
        if(linearRegression){
						getFirstFloat(line, &float_data, 1.0f, linearRegression);
				}
				else{
					getFirstFloat(line, &float_data, normaliseCeiling, linearRegression); // converts the first data value from the string line into a float
				}
#ifdef DEEPTEST
				printf("Float data value is %f\n", float_data);
#endif
				
				//=====INPUT NODE PROCESSING FIRST=========
				// Decorate the input with the data value
				if( linearRegression ){ 
					if(oneTime){
						superpopulation.miscpopulation[0]->inputLayer->input = (float) dataCount;
					}
					else{
						superpopulation.oldpopulation[member]->inputLayer->input = (float) dataCount; // x value becomes simply the counter
					}
					expectedResultRegression = float_data; // For regression the hoped-for result is the y value 
				}
				else{
					if(oneTime){
							superpopulation.miscpopulation[0]->inputLayer->input = float_data;
					}
					
					superpopulation.oldpopulation[member]->inputLayer->input = float_data;
				}
				
				
				if(oneTime){
					//Multiply input by weight
					superpopulation.miscpopulation[0]->inputLayer->output = superpopulation.miscpopulation[0]->inputLayer->input * superpopulation.miscpopulation[0]->inputLayer->weight;
					
					//Add the bias
					superpopulation.miscpopulation[0]->inputLayer->output += superpopulation.miscpopulation[0]->inputLayer->bias;
					
					//Run through the activation function
					superpopulation.miscpopulation[0]->inputLayer->output = processActivationFunction( superpopulation.miscpopulation[0]->inputLayer, superpopulation.miscpopulation[0]->inputLayer->output );
					
					
					
					//==========HIDDEN LAYER PROCESSING============
					//Now do the same for each hidden-layer node.....
					int h = 0;
					for(h = 0; h < superpopulation.miscpopulation[0]->numberOfHiddenNodes; h++){
						//decorate hidden node input from input node's output
						
						superpopulation.miscpopulation[0]->hiddenLayer[h]->input = superpopulation.miscpopulation[0]->inputLayer->output;
						// Multiply input by weight
						superpopulation.miscpopulation[0]->hiddenLayer[h]->output = superpopulation.miscpopulation[0]->hiddenLayer[h]->input * superpopulation.miscpopulation[0]->hiddenLayer[h]->weight;
						// Add in bias
						superpopulation.miscpopulation[0]->hiddenLayer[h]->output += superpopulation.miscpopulation[0]->hiddenLayer[h]->bias;
						
						//Run through activation function
						superpopulation.miscpopulation[0]->hiddenLayer[h]->output = processActivationFunction(superpopulation.miscpopulation[0]->hiddenLayer[h], superpopulation.miscpopulation[0]->hiddenLayer[h]->output); 
						

					}
				}
				else{
					//Multiply input by weight
					superpopulation.oldpopulation[member]->inputLayer->output = superpopulation.oldpopulation[member]->inputLayer->input * superpopulation.oldpopulation[member]->inputLayer->weight;
					
					//Add the bias
					superpopulation.oldpopulation[member]->inputLayer->output += superpopulation.oldpopulation[member]->inputLayer->bias;
					
					//Run through the activation function
					superpopulation.oldpopulation[member]->inputLayer->output = processActivationFunction( superpopulation.oldpopulation[member]->inputLayer, superpopulation.oldpopulation[member]->inputLayer->output);
					
				
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
						superpopulation.oldpopulation[member]->hiddenLayer[h]->output =  processActivationFunction(superpopulation.oldpopulation[member]->hiddenLayer[h],superpopulation.oldpopulation[member]->hiddenLayer[h]->output );
						
					}
				}

				//=======OUTPUT LAYER PROCESSING============
				if(oneTime){
					int o, w;
					if(linearRegression){
						superpopulation.miscpopulation[0]->outputLayer[0]->output = 0.0f; // clear output value 
						for(w = 0; w < superpopulation.miscpopulation[0]->numberOfHiddenNodes; w++){
										// input value (which is output of preceding layer node) * appropriate weight....
										superpopulation.miscpopulation[0]->outputLayer[0]->output += superpopulation.miscpopulation[0]->hiddenLayer[w]->output * superpopulation.miscpopulation[0]->outputLayer[0]->weights[w];
						}
						// Now add in bias
						superpopulation.miscpopulation[0]->outputLayer[0]->output += superpopulation.miscpopulation[0]->outputLayer[0]->bias;
							
						// Run through activation function
						superpopulation.miscpopulation[0]->outputLayer[0]->output = processActivationFunction( superpopulation.miscpopulation[0]->outputLayer[0], superpopulation.miscpopulation[0]->outputLayer[0]->output);
														
					}
					else{
						
						for(o = 0; o < superpopulation.miscpopulation[0]->numberOfOutputNodes; o++){
								superpopulation.miscpopulation[0]->outputLayer[o]->output = 0.0f; // clear output value 
								// Sum each result of multiplying input from preceding layer with appropriate weight
								for(w = 0; w < superpopulation.miscpopulation[0]->numberOfHiddenNodes; w++){
										// input value (which is output of preceding layer node) * appropriate weight....
										superpopulation.miscpopulation[0]->outputLayer[o]->output += superpopulation.miscpopulation[0]->hiddenLayer[w]->output * superpopulation.miscpopulation[0]->outputLayer[o]->weights[w];
								}
								
								// Now add in bias
								superpopulation.miscpopulation[0]->outputLayer[o]->output += superpopulation.miscpopulation[0]->outputLayer[o]->bias;
							
								// Run through activation function
								superpopulation.miscpopulation[0]->outputLayer[o]->output = processActivationFunction(	superpopulation.miscpopulation[0]->outputLayer[o],superpopulation.miscpopulation[0]->outputLayer[o]->output);
								
								if( !linearRegression){
												fprintf(fpW, "%f,",superpopulation.miscpopulation[0]->outputLayer[o]->output);
								}
						}
					}
					
					if(  (!linearRegression) ){ // Close the logged 3 outputs line that is being written
						fprintf(fpW,"\n"); // end that line by writing newline character
					}
					if( (linearRegression)  ){ // only use first output node for linear regression
							fprintf(fpW,"%f,\n",superpopulation.miscpopulation[0]->outputLayer[0]->output);
					}
				}
				else{
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
							superpopulation.oldpopulation[member]->outputLayer[o]->output = processActivationFunction( superpopulation.oldpopulation[member]->outputLayer[o], superpopulation.oldpopulation[member]->outputLayer[o]->output);
					}
				}
				
				if( oneTime != 1 ){ // if doing normal processing within algorithm rather than a one-off run-through 
					
						//=============FIND NORMALISED LMS of ANN AFTER PROCESSING THAT LINE OF DATA=====================
						if( linearRegression ){
							superpopulation.oldpopulation[member]->lms += normalisedLms_linearRegression( superpopulation.oldpopulation[member]->outputLayer[0]->output, expectedResultRegression) / dataCount;
						}
						else{
							superpopulation.oldpopulation[member]->lms += normalisedLms( superpopulation.oldpopulation[member]->outputLayer[0]->output, superpopulation.oldpopulation[member]->outputLayer[1]->output, superpopulation.oldpopulation[member]->outputLayer[2]->output, expectedResultLow, expectedResultMed, expectedResultHigh) / dataCount;
						}
#ifdef DEEPTEST
				printf("LMS is currently %f\n",superpopulation.oldpopulation[member]->lms);
#endif
				}

    }
     // Refer: https://stackoverflow.com/questions/3501338/c-read-file-line-by-line
		fclose(fp);
		if(oneTime){
			fclose(fpW); // as it was only opened for oneTime writing to
		}

		if( line ){
			free(line);
		}

   //exit(1); // for exiting in order to see result so far without iterating through all the data!!!
		
}

//This function simply prints the values currently contained in the fields of a Node struct
void printNode(Node* paramNode, int printWeightsArray, int numberHidden){
	if(paramNode->activationFunction == 1)
	  printf("Node has activation function sigmoid\n");
	if(paramNode->activationFunction == 2)
	  printf("Node has activation function relu\n");	
	if(paramNode->activationFunction == 3)
	  printf("Node has activation function cosh\n");
	  
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

void writeFFANNtoFile(Individual* citizen, int currentCycle){
		FILE* fp;
		fp = fopen("log.txt", "a");
		fprintf(fp, "======== Best ANN Found Details =========: \n");
		fprintf(fp, "===Cycle is %d===\n",currentCycle);
		fprintf(fp, "Input layer: \n");
		if(citizen->inputLayer->activationFunction == 1)
			fprintf(fp, "Activation Function: sigmoid\n");
		if(citizen->inputLayer->activationFunction == 2)
			fprintf(fp, "Activation Function: relu\n");
		if(citizen->inputLayer->activationFunction == 3)
			fprintf(fp, "Activation Function: cosh\n");
		fprintf(fp, "Input weight: %f, input bias: %f \n",citizen->inputLayer->weight, citizen->inputLayer->bias);
		fprintf(fp, "Input weight: %f, input bias: %f \n",citizen->inputLayer->weight, citizen->inputLayer->bias);
		int c,d; 
		for(c=0; c < citizen->numberOfHiddenNodes; c++){
			fprintf(fp, "\nHidden layer node %d:- weight: %f, bias: %f\n",c, citizen->hiddenLayer[c]->weight, citizen->hiddenLayer[c]->bias);
					if(citizen->hiddenLayer[c]->activationFunction == 1)
						fprintf(fp, "Activation Function: sigmoid\n");
					if(citizen->hiddenLayer[c]->activationFunction == 2)
						fprintf(fp, "Activation Function: relu\n");
					if(citizen->hiddenLayer[c]->activationFunction == 3)
						fprintf(fp, "Activation Function: cosh\n");
		}
		
		for(c=0; c < citizen->numberOfOutputNodes; c++){
				fprintf(fp, "\nOutput layer node %d :-  bias: %f \n", c, citizen->outputLayer[c]->bias);
							if(citizen->outputLayer[c]->activationFunction == 1)
									fprintf(fp, "Activation Function: sigmoid\n");
							if(citizen->outputLayer[c]->activationFunction == 2)
									fprintf(fp, "Activation Function: relu\n");
							if(citizen->outputLayer[c]->activationFunction == 3)
									fprintf(fp, "Activation Function: cosh\n");
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
  
#ifdef TEST
printf("length of hidden layer inside constructFFANN : %d \n",numberOfHidden);
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
	int index5, index6, index7, index8;
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
	
	index5 = getRandomIndividualIndex(); 
	while( (index5 == index4) || (index5 == index3) || (index5 == index2) || (index5 == index1) ){
		index5 = getRandomIndividualIndex();
	}
	
	index6 = getRandomIndividualIndex(); 
	while( (index6 == index5) || (index6 == index4) || (index6 == index3) || (index6 == index2) || (index6 == index1) ){
		index6 = getRandomIndividualIndex();
	}
	
	index7 = getRandomIndividualIndex(); 
	while( (index7 == index6) || (index7 == index5) || (index7 == index4) || (index7 == index3) || (index7 == index2) || (index7 == index1) ){
		index7 = getRandomIndividualIndex();
	}
	
	index8 = getRandomIndividualIndex(); 
	while( (index8 == index7) || (index8 == index6) || (index8 == index5) || (index8 == index4) || (index8 == index3) || (index8 == index2) || (index8 == index1) ){
		index8 = getRandomIndividualIndex();
	}
	
	Individual** tournArray = (Individual**) malloc( individualSizeMemory * tournamentSize );
	
	#ifdef AUXTEST
	printf("Chosen tournament member indexes are %d %d %d %d %d %d %d %d\n",index1, index2, index3, index4, index5, index6, index7, index8);
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
	
	#ifdef AUXTEST
	printf("About to copy indiv at index %d into tournArray[4]\n",index5);
	#endif
	
	tournArray[4] = &indTourn4;
	copyIndividual( superpopulation->oldpopulation[index5], tournArray[4]);
	
	#ifdef AUXTEST
	printf("About to copy indiv at index %d into tournArray[5]\n",index6);
	#endif
	
	tournArray[5] = &indTourn5;
	copyIndividual( superpopulation->oldpopulation[index6], tournArray[5]);
	
	#ifdef AUXTEST
	printf("About to copy indiv at index %d into tournArray[6]\n",index7);
	#endif
	
	tournArray[6] = &indTourn6;
	copyIndividual( superpopulation->oldpopulation[index7], tournArray[6]);
	
	#ifdef AUXTEST
	printf("About to copy indiv at index %d into tournArray[7]\n",index8);
	#endif
	
	tournArray[7] = &indTourn7;
	copyIndividual( superpopulation->oldpopulation[index8], tournArray[7]);
	
	bubbleSort(tournArray, tournamentSize); 

#ifdef AUXTEST
	printf("sorted tournarray is: \n");
	printFFANN(tournArray[0]);
	printFFANN(tournArray[1]);
	printFFANN(tournArray[2]);
	printFFANN(tournArray[3]);
	printFFANN(tournArray[4]);
	printFFANN(tournArray[5]);
	printFFANN(tournArray[6]);
	printFFANN(tournArray[7]);
#endif
	
	// Now do breeding with probability, e.g. just take one of the best 2 or with prob do breeding between best 2 of tournament
	float prob = getRandomBiasValueFloat(1.0f, -2.0f); // will get a positive or negative value
	if(prob > 0.0f){ //33% chance of just copying individual as is... very boring thing to happen
		// tournArray[0] contains the individual with the best lms, e.g. the lowest, due to bubblesort!!!
		copyIndividual(tournArray[0], superpopulation->newpopulation[newpopMemberIndex]); // just copy a parent to new generation
		printf("Just copied tournament winner individual to new generation as is without breeding\n");
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
