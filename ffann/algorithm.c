
#include <omp.h>

#include "algorithm_aux.c"


#ifndef MATH_H
#define MATH_H 1
#include<math.h>
#endif


int main(int argc, char * argv[]){
	
	//printf("Run this like /.new.exe a f"); //where a specifies accelerometer data, f specifies fft data
	srand((unsigned int)time(NULL)); // needed for random number generation...
	
	initialiseVariables(); // 
#ifdef TEST
	printf("Just initialised variables\n");
#endif


	constructPopulation(&superpopulation);

#ifdef TEST
	printf("Just constructed population struct\n");
#endif

    addIndividualsToPopulations(&superpopulation); // also sets the pointers for all input, hidden and output nodes
    
#ifdef TEST
	printf("Just added invididuals to oldpopulation\n");
#endif    

	strcpy( linearDataFilename, "lineardata.data");
	strcpy(filenameWrite,"oneTimeOutput.data");

	if(argc > 1){
		if( !strcmp(argv[1],"a") ){
			
	#ifdef TEST
			printf("Accelerometer data chosen\n");
	#endif
			chosenSensor = 0; // 0 is for accelerometer, 1 is for lidar. 
			normaliseCeiling = 300.0f; // based on empirical evidence... 
			if(argc > 2){
				if( !strcmp(argv[2],"f") ){
		#ifdef TEST
				printf("FFT data chosen\n");
		#endif			
				
					normaliseCeiling = 250.0f; // based on empirical evidence for accel fft data
					strcpy(filenamesListLow[0],"fftAccel15psi.data"); strcpy(filenamesListLow[1],"fftAccel20psi.data"); strcpy(filenamesListLow[2],"fftAccel25psi.data");
					strcpy(filenamesListMiddle[0],"fftAccel30psi.data"); strcpy(filenamesListMiddle[1],"fftAccel35psi.data"); strcpy(filenamesListMiddle[2],"fftAccel40psi.data"); strcpy(filenamesListMiddle[3],"fftAccel45psi.data");
					strcpy(filenamesListHigh[0],"fftAccel50psi.data"); strcpy(filenamesListHigh[1],"fftAccel55psi.data"); strcpy(filenamesListHigh[2],"fftAccel60psi.data");

				}
				else if( !strcmp(argv[2],"c") ){
		#ifdef TEST
				printf("Cepstrogram data chosen\n");
		#endif		
					normaliseCeiling = 50000.0f; // based on empirical evidence for accel cepstrogram data
					strcpy(filenamesListLow[0],"cepAccel15psi.data"); strcpy(filenamesListLow[1],"cepAccel20psi.data"); strcpy(filenamesListLow[2],"cepAccel25psi.data");
					strcpy(filenamesListMiddle[0],"cepAccel30psi.data"); strcpy(filenamesListMiddle[1],"cepAccel35psi.data"); strcpy(filenamesListMiddle[2],"cepAccel40psi.data"); strcpy(filenamesListMiddle[3],"cepAccel45psi.data");
					strcpy(filenamesListHigh[0],"cepAccel50psi.data"); strcpy(filenamesListHigh[1],"cepAccel55psi.data"); strcpy(filenamesListHigh[2],"cepAccel60psi.data");

				}
				else if ( !strcmp(argv[2],"r") ){
		#ifdef TEST
				printf("Running average accelerometer data chosen\n");
		#endif		
					normaliseCeiling = 300.0f; // from empirical evidence looking at the largest datafile values
					strcpy(filenamesListLow[0],"ra_Accel15psi.data"); strcpy(filenamesListLow[1],"ra_Accel20psi.data"); strcpy(filenamesListLow[2],"ra_Accel25psi.data");
					strcpy(filenamesListMiddle[0],"ra_Accel30psi.data"); strcpy(filenamesListMiddle[1],"ra_Accel35psi.data"); strcpy(filenamesListMiddle[2],"ra_Accel40psi.data"); strcpy(filenamesListMiddle[3],"ra_Accel45psi.data");
					strcpy(filenamesListHigh[0],"ra_Accel50psi.data"); strcpy(filenamesListHigh[1],"ra_Accel55psi.data"); strcpy(filenamesListHigh[2],"ra_Accel60psi.data");

				}
				else if ( !strcmp(argv[2],"d") ){
		#ifdef TEST
				printf("Difference accelerometer data chosen\n"); // squares of differences between consecutive raw data values
		#endif		
					normaliseCeiling = 300.0f; // from empirical evidence looking at the largest datafile values
					strcpy(filenamesListLow[0],"ra_Accel15psi.data"); strcpy(filenamesListLow[1],"ra_Accel20psi.data"); strcpy(filenamesListLow[2],"ra_Accel25psi.data");
					strcpy(filenamesListMiddle[0],"ra_Accel30psi.data"); strcpy(filenamesListMiddle[1],"ra_Accel35psi.data"); strcpy(filenamesListMiddle[2],"ra_Accel40psi.data"); strcpy(filenamesListMiddle[3],"ra_Accel45psi.data");
					strcpy(filenamesListHigh[0],"ra_Accel50psi.data"); strcpy(filenamesListHigh[1],"ra_Accel55psi.data"); strcpy(filenamesListHigh[2],"ra_Accel60psi.data");

				}
				else{
					printf("The argument after a needs to be either f for fft, or c for cepstrogram processed data...\n");
					return 1;
				}
			}
			else{ // raw accelerometer data
					strcpy(filenamesListLow[0],"Accel15psi.data"); strcpy(filenamesListLow[1],"Accel20psi.data"); strcpy(filenamesListLow[2],"Accel25psi.data");
					strcpy(filenamesListMiddle[0],"Accel30psi.data"); strcpy(filenamesListMiddle[1],"Accel35psi.data"); strcpy(filenamesListMiddle[2],"Accel40psi.data"); strcpy(filenamesListMiddle[3],"Accel45psi.data");
					strcpy(filenamesListHigh[0],"Accel50psi.data"); strcpy(filenamesListHigh[1],"Accel55psi.data"); strcpy(filenamesListHigh[2],"Accel60psi.data");
					
			}
			if(argc > 3){
				if( !strcmp(argv[3],"n") ){ // choosing to normalise
		#ifdef TEST
				printf("Normalisation will be done when reading in data\n");
		#endif	
				
					normalise = 1; // by default it is 0 for triclassification
				}
			}
		}
		else if( !strcmp(argv[1],"l") ){
	#ifdef TEST
			printf("Lidar data chosen\n");
	#endif
			chosenSensor = 1; // 1 for lidar
			normaliseCeiling = 500.0f; // based on empirical evidence...
			strcpy(filenamesListLow[0],"Lidar15psi.data"); strcpy(filenamesListLow[1],"Lidar20psi.data"); strcpy(filenamesListLow[2],"Lidar25psi.data");
			strcpy(filenamesListMiddle[0],"Lidar30psi.data"); strcpy(filenamesListMiddle[1],"Lidar35psi.data"); strcpy(filenamesListMiddle[2],"Lidar40psi.data"); strcpy(filenamesListMiddle[3],"Lidar45psi.data");
			strcpy(filenamesListHigh[0],"Lidar50psi.data"); strcpy(filenamesListHigh[1],"Lidar55psi.data"); strcpy(filenamesListHigh[2],"Lidar60psi.data");
		}
		else if ( !strcmp(argv[1],"r") ){ // if regression specified
			
				printf("Linear regression chosen\n");
				linR = 1;
				strcpy(linearDataFilename,"lineardata.data");
				
		}
		else if ( !strcmp(argv[1],"c") ){ // if classification specified.... 
			
				// expectedPressure is first argument, 0 = low, 1 = med, 2 = high
				classify( 0, "dataToClassify.data", "", 0, 0, 1.0f, 0.0f, 0.0f, 300.0f, 1  );

		}
		else{
				printf("You need to specify a for accel, l for lidar, or r for regression as the first argument\n");
				return 1;
		}
	}
	else{
		printf("You need to specify a for accel, l for lidar, or r for regression as the first argument when running this script\n");
		return 1;
	}
	
	/*if(argc == 4){
		if( !strcmp(argv[3],"r") ){ // if regression specified
			printf("Linear regression chosen\n");
			linR = 1;
			strcpy(linearDataFilename,"lineardata.data");
		}
	}*/
	
	/*if(argc == 5){
			if( !strcmp(argv[4],"o") ){ // if one-time run through specified
				printf("One-time run-through chosen\n");
				oneTime = 1;
			}
	}*/
	
#ifdef TEST	
	printf("Just about to populate filenamesLists\n");
#endif	
	filenamesList[0][0] = filenamesListLow[0]; filenamesList[0][1] = filenamesListLow[1]; filenamesList[0][2] = filenamesListLow[2];
	filenamesList[1][0] = filenamesListMiddle[0]; filenamesList[1][1] = filenamesListMiddle[1]; filenamesList[1][2] = filenamesListMiddle[2]; filenamesList[1][3] = filenamesListMiddle[3];
	filenamesList[2][0] = filenamesListHigh[0]; filenamesList[2][1] = filenamesListHigh[1]; filenamesList[2][2] = filenamesListHigh[2];
#ifdef TEST
	printf("Just after filenamesLists populated\n");
#endif

	
	constructFFANN(superpopulation.miscpopulation, 0); // create initial best individual
	constructFFANN(superpopulation.miscpopulation, 1); // create initial sorting individual
	constructFFANN(superpopulation.miscpopulation, 2); // create initial tournament0 individual
	constructFFANN(superpopulation.miscpopulation, 3); // create initial tournament1 individual
	constructFFANN(superpopulation.miscpopulation, 4); // create initial tournament2 individual
	constructFFANN(superpopulation.miscpopulation, 5); // create initial tournament3 individual
	constructFFANN(superpopulation.miscpopulation, 6); // create initial tournament4 individual
	constructFFANN(superpopulation.miscpopulation, 7); // create initial tournament5 individual
	constructFFANN(superpopulation.miscpopulation, 8); // create initial tournament6 individual
	constructFFANN(superpopulation.miscpopulation, 9); // create initial tournament7 individual

	int c = 0;
/*	for( c=0; c < numCycles; c++){*/
	for(c=0; c < popsize; c++){
		 constructFFANN(superpopulation.oldpopulation,c); // # create initial old population citizen empty shells 
		 constructFFANN(superpopulation.newpopulation,c); // create initial new population citizen empty shells
		 //printFFANN(superpopulation.oldpopulation[c]);
	 }

	 /*
	  * ========expectedResult_Explanation================ 
	  * eL = 1.0f; eM = 0.0f; eH = 0.0f # LOW pressure data being trained with, note there are 3 data files
        eL = 0.0f; eM = 0.0f; eH = 0.0f # MED pressure data being trained with, note there are 4 data files
        eL = 0.0f; eM = 0.0f; eH = 1.0f # HIGH pressure data being trained with, note there are 3 data files
	  * ==========endOfExpectedResult_Explanation=========
	  */

	float eL; float eM; float eH;
	int m = 0;
	for(c=0;c<numCycles; c++){// # number of cycles of this evolutionary algorithm
		//#Process the input data through each population member
		printf("cycle is %d\n",c); //+str(t)+", just about to process member "+str(x))    

		#pragma omp parallel for num_threads(4)
			for(m=0; m < popsize; m++){ // #e.g. for each member FFANN, process it
				printf("Just about to process member %d\n",m);
				superpopulation.oldpopulation[m]->lms = 0.0f; // reset this just before processing so that a fresh lms can be calculated
				
				eL = expectedResultTriClassification; eM = 0.0f; eH = 0.0f; // low pressure is expected result

				if( linR ){ // process linear data file for regression solving
						process(linearDataFilename, filenameWrite, m, linR, eL, eM, eH, normaliseCeiling, oneTime); 
				}
				else{ // process all the different data files for multiple output solving
					process(filenamesList[0][0], filenameWrite, m, linR, eL, eM, eH, normaliseCeiling, oneTime); //process selected data through an ANN
					process(filenamesList[0][1], filenameWrite, m, linR, eL, eM, eH, normaliseCeiling, oneTime); //process selected data through an ANN
					process(filenamesList[0][2], filenameWrite, m, linR, eL, eM, eH, normaliseCeiling, oneTime);//process selected data through an ANN
					
					eL = 0.0f; eM = expectedResultTriClassification; eH = 0.0f; // medium pressure is expected result
					process(filenamesList[1][0], filenameWrite, m, linR, eL, eM, eH, normaliseCeiling, oneTime); //process selected data through an ANN
					process(filenamesList[1][1], filenameWrite, m, linR, eL, eM, eH, normaliseCeiling, oneTime); //process selected data through an ANN
					process(filenamesList[1][2], filenameWrite, m, linR, eL, eM, eH, normaliseCeiling, oneTime); //process selected data through an ANN
					process(filenamesList[1][3], filenameWrite, m, linR, eL, eM, eH, normaliseCeiling, oneTime); //process selected data through an ANN
					
					eL = 0.0f; eM = 0.0f; eH = expectedResultTriClassification; // high pressure is expected result
					process(filenamesList[2][0], filenameWrite, m, linR, eL, eM, eH, normaliseCeiling, oneTime); //process selected data through an ANN
					process(filenamesList[2][1], filenameWrite, m, linR, eL, eM, eH, normaliseCeiling, oneTime); //process selected data through an ANN
					process(filenamesList[2][2], filenameWrite, m, linR, eL, eM, eH, normaliseCeiling, oneTime); //process selected data through an ANN
				}
				
				// If the ANN just processed has a better LMS than that found so far... copy it to the best ANN Individual 
				#pragma omp critical
				{
					if( superpopulation.oldpopulation[m]->lms < bestlms ){
							bestlms = superpopulation.oldpopulation[m]->lms; // update current best LMS
							printf("===========New best ANN found: ===========\n");
							
							//====================SAVE BEST DETAILS================
							copyIndividual(superpopulation.oldpopulation[m], superpopulation.miscpopulation[0]); // copy new best to Best Individual 
							
							printFFANN( superpopulation.miscpopulation[0] );
							// Now also write the best ANN to file
							writeFFANNtoFile( superpopulation.miscpopulation[0], c );
					}
				} // end of pragma omp critical 
			}
		
		//=============CREATE NEW GENERATION 
		
		//First Elitism
		copyIndividual(&indX, superpopulation.newpopulation[0] );
		//m now starts at 1 as elitism took care of index 0... 
		
		//Now Tournament Selection and breeding
		for(m=1; m < popsize; m++){
			printf("Adding member %d to newpopulation\n",m);
			// Tournament selection and breeding
			tournament(&superpopulation, m);
			
			// Probability of mutation 
			float prob = getRandomBiasValueFloat(2.0f, -2.0f);
			if( prob < 0.0f ){ //50% chance of mutation
#ifdef TEST
				printf("ANN before mutation is :\n");
				printFFANN( superpopulation.newpopulation[m] );
#endif
				mutate( superpopulation.newpopulation[m] );
#ifdef TEST
				printf("ANN after mutation is :\n");
				printFFANN( superpopulation.newpopulation[m] );
#endif
			}
			superpopulation.newpopulation[m]->lms = 0.0f; // need to reset this as breeding and mutation will change the lms score
		}
		
		// Finally copy the new generation into the old generation array
		int c = 0;
		for(c=0; c < popsize; c++){
				copyIndividual(superpopulation.newpopulation[c], superpopulation.oldpopulation[c]);
		}
 	} // end of cycles loop
 	
 	printf("Algorithm complete, please see log.txt for details of best ANN found\n");
 	
 	if(linR){
		oneTime = 1; // setting this for a one-time run-through of data throug best ANN and logging of outputs
		superpopulation.miscpopulation[0]->lms = 0.0f; // Resetting lms before the processing happens
		processBestOneTime(linearDataFilename, filenameWrite, 0, linR, eL, eM, eH, normaliseCeiling, oneTime); 
		printf("Also see %s for details of one-time run-through output of best ANN\n",filenameWrite);
	}
	else{
			oneTime = 1; // setting this for a one-time run-through of data throug best ANN and logging of outputs
			if( oneTime ){ // This shouldn't be necessary  but I had a bug where the first few lines would process as though !oneTime
				superpopulation.miscpopulation[0]->lms = 0.0f; // Resetting lms before the processing happens
				
				eL = expectedResultTriClassification; eM = 0.0f; eH = 0.0f;
				processBestOneTime(filenamesList[0][0], filenameWrite, m, linR, eL, eM, eH, normaliseCeiling, oneTime); //process selected data through an ANN
				processBestOneTime(filenamesList[0][1], filenameWrite, m, linR, eL, eM, eH, normaliseCeiling, oneTime); //process selected data through an ANN
				processBestOneTime(filenamesList[0][2], filenameWrite, m, linR, eL, eM, eH, normaliseCeiling, oneTime);//process selected data through an ANN
				
				eL = 0.0f; eM = expectedResultTriClassification; eH = 0.0f; // medium pressure is expected result
				processBestOneTime(filenamesList[1][0], filenameWrite, m, linR, eL, eM, eH, normaliseCeiling, oneTime); //process selected data through an ANN
				processBestOneTime(filenamesList[1][1], filenameWrite, m, linR, eL, eM, eH, normaliseCeiling, oneTime); //process selected data through an ANN
				processBestOneTime(filenamesList[1][2], filenameWrite, m, linR, eL, eM, eH, normaliseCeiling, oneTime); //process selected data through an ANN
				processBestOneTime(filenamesList[1][3], filenameWrite, m, linR, eL, eM, eH, normaliseCeiling, oneTime); //process selected data through an ANN
				
				eL = 0.0f; eM = 0.0f; eH = expectedResultTriClassification; // high pressure is expected result
				processBestOneTime(filenamesList[2][0], filenameWrite, m, linR, eL, eM, eH, normaliseCeiling, oneTime); //process selected data through an ANN
				processBestOneTime(filenamesList[2][1], filenameWrite, m, linR, eL, eM, eH, normaliseCeiling, oneTime); //process selected data through an ANN
				processBestOneTime(filenamesList[2][2], filenameWrite, m, linR, eL, eM, eH, normaliseCeiling, oneTime); //process selected data through an ANN

				printf("Also see %s for details of one-time run-through output of best ANN\n",filenameWrite);
				
			}
	}

#ifdef TEST
	for(c=0; c < popsize; c++){
		 printf("Now back in main ... to print the freshly minted citizen.....\n");
		 printFFANN(superpopulation.oldpopulation[c]); 
	 }
	 
	 printf("NOW TESTING COPY FUNCTIONALITY\n");
	 copyIndividual(superpopulation.oldpopulation[0], superpopulation.newpopulation[0]);
	 printFFANN( superpopulation.oldpopulation[0] );
	 printFFANN( superpopulation.newpopulation[0] );
#endif

	freeAllocatedMemory(); // FINALLY CALL FREE_MEMORY FUNCTION WHICH DEALLOCATES THE MALLOCs :) 

	return 0;

} // end of main

