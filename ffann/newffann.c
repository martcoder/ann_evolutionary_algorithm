

#include "newaux.c"


#ifndef MATH_H
#define MATH_H 1
#include<math.h>
#endif




//from ffann_aux import * #imports global variables, Node and Individual classes


int main(int argc, char * argv[]){
	
	//printf("Run this like /.new.exe a f"); //where a specifies accelerometer data, f specifies fft data
	srand((unsigned int)time(NULL)); // needed for random number generation...
	
	initialiseVariables(); // 
#ifdef TEST
	printf("Just initialised variables\n");
#endif
	/*constructNode( &bestInputLayer );
	bestHiddenLayer = (Node *) malloc( nodeSizeMemory * hiddenMax );
	bestOutputLayer = (Node *) malloc( nodeSizeMemory * outputLayerLength );
	* */
	//constructNode( &inputLayer ); // create inputLayer node
	
	//hiddenLayer = (Node *) malloc( nodeSizeMemory * hiddenMax  );

	//outputLayer = (Node *) malloc( nodeSizeMemory * outputLayerLength  );

	constructPopulation(&superpopulation);

#ifdef TEST
	printf("Just constructed population struct\n");
#endif

    addIndividualsToPopulations(&superpopulation); // also sets the pointers for all input, hidden and output nodes
    
#ifdef TEST
	printf("Just added invididuals to oldpopulation\n");
#endif    


	if( !strcmp(argv[1],"a") ){
		
#ifdef TEST
		printf("Accelerometer data chosen\n");
#endif
		chosenSensor = 0; // 0 is for accelerometer, 1 is for lidar. 
		normaliseCeiling = 300.0f; // based on empirical evidence... 
		if( !strcmp(argv[2],"f") ){
#ifdef TEST
		printf("FFT data chosen\n");
#endif			
		
			normaliseCeiling = 250.0f; // based on empirical evidence for accel fft data
			strcpy(filenamesListLow[0],"fftAccel15psi.data"); strcpy(filenamesListLow[1],"fftAccel20psi.data"); strcpy(filenamesListLow[2],"fftAccel25psi.data");
			strcpy(filenamesListMiddle[0],"fftAccel30psi.data"); strcpy(filenamesListMiddle[1],"fftAccel35psi.data"); strcpy(filenamesListMiddle[2],"fftAccel40psi.data"); strcpy(filenamesListMiddle[3],"fftAccel45psi.data");
			strcpy(filenamesListHigh[0],"fftAccel50psi.data"); strcpy(filenamesListHigh[1],"fftAccel55psi.data"); strcpy(filenamesListHigh[2],"fftAccel60psi.data");

		}
		else{
			strcpy(filenamesListLow[0],"Accel15psi.data"); strcpy(filenamesListLow[1],"Accel20psi.data"); strcpy(filenamesListLow[2],"Accel25psi.data");
			strcpy(filenamesListMiddle[0],"Accel30psi.data"); strcpy(filenamesListMiddle[1],"Accel35psi.data"); strcpy(filenamesListMiddle[2],"Accel40psi.data"); strcpy(filenamesListMiddle[3],"Accel45psi.data");
			strcpy(filenamesListHigh[0],"Accel50psi.data"); strcpy(filenamesListHigh[1],"Accel55psi.data"); strcpy(filenamesListHigh[2],"Accel60psi.data");

			
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
	else{
		printf("You need to specify accel or lidar as the argument when running this script");
		return 1;
	}
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
	float bestMemberLMSIndex = 1000.0f;
	for(c=0;c<numCycles; c++){// # number of cycles of this evolutionary algorithm
		//#Process the input data through each population member
		printf("cycle is %d\n",c); //+str(t)+", just about to process member "+str(x))    

		for(m=0; m < popsize; m++){ // #e.g. for each member FFANN, process it
			printf("Just about to process member %d\n",m);
			superpopulation.oldpopulation[m]->lms = 0.0f; // reset this just before processing
			//#print("length of hidden layer is "+str(len(hiddenLayer)))
			//#Run through each line of data in each datafile
			eL = 1.0f; eM = 0.0f; eH = 0.0f; // low pressure is expected result
			process(filenamesList[0][0], numberOfLowDataFiles, m, eL, eM, eH, normaliseCeiling); //process selected data through an ANN
			process(filenamesList[0][1], numberOfLowDataFiles, m, eL, eM, eH, normaliseCeiling); //process selected data through an ANN
			process(filenamesList[0][2], numberOfLowDataFiles, m, eL, eM, eH, normaliseCeiling);//process selected data through an ANN
			
			eL = 0.0f; eM = 1.0f; eH = 0.0f; // medium pressure is expected result
			process(filenamesList[1][0], numberOfMedDataFiles, m, eL, eM, eH, normaliseCeiling); //process selected data through an ANN
			process(filenamesList[1][1], numberOfMedDataFiles, m, eL, eM, eH, normaliseCeiling); //process selected data through an ANN
			process(filenamesList[1][2], numberOfMedDataFiles, m, eL, eM, eH, normaliseCeiling); //process selected data through an ANN
			process(filenamesList[1][3], numberOfMedDataFiles, m, eL, eM, eH, normaliseCeiling); //process selected data through an ANN
			
			eL = 0.0f; eM = 0.0f; eH = 1.0f; // high pressure is expected result
			process(filenamesList[2][0], numberOfHighDataFiles, m, eL, eM, eH, normaliseCeiling); //process selected data through an ANN
			process(filenamesList[2][1], numberOfHighDataFiles, m, eL, eM, eH, normaliseCeiling); //process selected data through an ANN
			process(filenamesList[2][2], numberOfHighDataFiles, m, eL, eM, eH, normaliseCeiling); //process selected data through an ANN
			
			// If the ANN just processed has a better LMS than that found so far... copy it to the best ANN Individual 
			if( superpopulation.oldpopulation[m]->lms < bestMemberLMSIndex ){
					bestMemberLMSIndex = superpopulation.oldpopulation[m]->lms; // update current best LMS
					printf("===========New best ANN found: ===========\n");
					copyIndividual(superpopulation.oldpopulation[m], superpopulation.miscpopulation[0]); // copy new best to Best Individual 
					
					printFFANN( superpopulation.miscpopulation[0] );
					// Now also write the best ANN to file
					writeFFANNtoFile( superpopulation.miscpopulation[0] );
			}
		}
		
		//====================SAVE BEST DETAILS================
		
		
		
		
		
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

/*
			#Get datafile result as LeastMeanSquared
			lmsavg = statistics.mean(lmsResult)
			#print("lmssum is "+str(lmssum))
			if lmsavg < bestlms: #Set this ffann as the best so far....
			  print("Found new best lms of "+str(lmsavg))
			  #meanResult = statistics.mean(result)
			  #print("And mean output was "+str(meanResult) )
			  #meanLOW = statistics.mean(resultLOW)
			  #meanMED = statistics.mean(resultMED)
			  #meanHIGH = statistics.mean(resultHIGH)
			  #print("mean LOW was "+str(meanLOW)+", mean MED was "+str(meanMED)+", mean HIGH was "+str(meanHIGH))
			  bestInputLayer = copy.deepcopy(global_population.oldpopulation[x].inputLayer)
			  #bestInputLayer.meanOutput = meanResult
			  #bestInputLayer.meanOutputLOW = meanLOW
			  #bestInputLayer.meanOutputMED = meanMED
			  #bestInputLayer.meanOutputHIGH = meanHIGH
			  #print("len of hidden layer is "+str(len(hiddenLayer)))
			  bestHiddenLayer = copy.deepcopy(global_population.oldpopulation[x].hiddenLayer)
			  bestOutputLayer = copy.deepcopy(global_population.oldpopulation[x].outputLayer)
			  bestlms = lmsavg
			lmsResult.clear()  # empty this ready for the next FFANN
			result = []
			resultMED = []
			resultLOW = []
			resultHIGH = []
			#lmssum = 0.0
			#Now make new ffann....cleaning up the previous one
			inputLayer = Node()
			hiddenLayer = []
			hiddenLayer.clear()
			outputLayer = []
			outputLayer.clear()

		  #Now CREATE NEW POPULATION

		  global_population.newpopulation = []
		  #Firstly do elitism
		  countElite = 0
		  for x in range(popsize):
			print("about to add memer to newpop")
			if countElite < elitism: #elitism num defined at beginning of script
			  addElite()
			  countElite += 1
			  print("Added an elite, total elite to add is "+str(elitism))
			else:
			  if len(global_population.newpopulation) < popsize:
				print("about to tournament")
				tournament() #to construct new population member
		  countElite = 0
		  #oldpopulation.clear()
		  
		  global_population.oldpopulation = []
		  print("New population contains "+str(len(global_population.newpopulation))+" members")
		  print("Old pop has been cleared and contains "+str(len(global_population.oldpopulation))+" members. Now copying new to old")
		  #for i in range(len(newpopulation)):
		  global_population.oldpopulation = copy.deepcopy(global_population.newpopulation) # now copy new population to old population
		  global_population.newpopulation = [] #.clear()
		  #newpopulation.clear()
		#Finally print and save the best FFANN....
		  print("New pop copied and cleared, now has "+str(len(global_population.newpopulation))+" members. Current population (old) contains "+str(len(global_population.oldpopulation))+" individuals\n")

		  print("..for example oldpop first node has input weight of "+str(global_population.oldpopulation[0].inputLayer.weight) )
		  print("..and oldpop first node has input value of "+str(global_population.oldpopulation[0].inputLayer.input) )

		  writer = open(str(datetime.now())+"_"+str(intendedResult)+".log","a")
		  writer.write("The best FFANN for "+str(intendedResult)+" with an lms of "+str(bestInputLayer.lms)+" is:\n")

		  writer.write("Input weight of "+str(bestInputLayer.weight)+" and bias is "+str(bestInputLayer.bias)+"\n")
		  for x in bestHiddenLayer:
		   writer.write("Hidden layer node, weight is "+str(x.weight)+" and bias is "+str(x.bias)+"\n")
		  #for m in bestOutputLayer:
		  for x in range(3):
			for w in bestOutputLayer[x].weights:
			  writer.write("Best output layer Node "+str(x)+" weight: "+str(w)+"\n ")
			writer.write("output bias for node "+str(x)+" is "+str(bestOutputLayer[x].bias))
		  writer.close()

		  print("The best FFANN for "+str(intendedResult)+" with an lms of "+str(bestInputLayer.lms)+" is:\n")

		  print("Input weight of "+str(bestInputLayer.weight)+" and bias is "+str(bestInputLayer.bias)+"\n")
		  for x in bestHiddenLayer:
		   print("Hidden layer node, weight is "+str(x.weight)+" and bias is "+str(x.bias)+"\n")

		  #for e in bestOutputLayer:
		  i = 0
		  for x in bestOutputLayer:
			for w in x.weights:
			  print("Best output layer Node "+str(i)+" weight is "+str(w)+"\n ")
			i = i + 1
			print("output bias is "+str(x.bias))
			*/
  /*}*/
}

