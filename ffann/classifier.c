//So gonna pass in each data file one by one, so the iteration over different data will happen in main
void classify( char * filename, char * filenameWrite, int member, int linearRegression, float expectedResultLow, float expectedResultMed, float expectedResultHigh, float normaliseCeiling, int oneTime  ){
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
