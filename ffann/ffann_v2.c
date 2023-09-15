
//from ffann_aux import * #imports global variables, Node and Individual classes


int main(int argc, char ** argv){
	
	srand((unsigned int)time(NULL)); // needed for random number generation...
	
	initialiseVariables(); // 
	constructNode( &bestInputLayer );
	bestHiddenLayer = (Node *) malloc( nodeSizeMemory * hiddenMax );
	bestOutputLayer = (Node *) malloc( nodeSizeMemory * outputLength );
	constructNode( &inputLayer ); // create inputLayer node
	
	hiddenLayer = (Node *) malloc( nodeSizeMemory * hiddenMax  );

	outputLayer = (Node *) malloc( nodeSizeMemory * outputLength  );

	constructPopulation(&population);

	if(argv[1] == 'a'){
		chosenSensor = 0; // 0 is for accelerometer, 1 is for lidar. 
		strcpy(filenamesListLow[0],"Accel15psi.data"); strcpy(filenamesListLow[1],"Accel20psi.data"); strcpy(filenamesListLow[2],"Accel25psi.data");
		strcpy(filenamesListMiddle[0],"Accel30psi.data"); strcpy(filenamesListMiddle[1],"Accel35psi.data"); strcpy(filenamesListMiddle[2],"Accel40psi.data"); strcpy(filenamesListMiddle[3],"Accel45psi.data");
		strcpy(filenamesListHigh[0],"Accel50psi.data"); strcpy(filenamesListHigh[1],"Accel55psi.data"); strcpy(filenamesListHigh[2],"Accel60psi.data");
	}
	else if( argv[1] == 'l'){
		chosenSensor = 1; // 1 for lidar
		strcpy(filenamesListLow[0],"Lidar15psi.data"); strcpy(filenamesListLow[1],"Lidar20psi.data"); strcpy(filenamesListLow[2],"Lidar25psi.data");
		strcpy(filenamesListMiddle[0],"Lidar30psi.data"); strcpy(filenamesListMiddle[1],"Lidar35psi.data"); strcpy(filenamesListMiddle[2],"Lidar40psi.data"); strcpy(filenamesListMiddle[3],"Lidar45psi.data");
		strcpy(filenamesListHigh[0],"Lidar50psi.data"); strcpy(filenamesListHigh[1],"Lidar55psi.data"); strcpy(filenamesListHigh[2],"Lidar60psi.data");
	}
	else{
		printf("You need to specify accel or lidar as the argument when running this script");
		return 1;
	}
	
	filenamesList[0] = filenamesListLow;
	filenamesList[1] = filenamesListMiddle;
	filenamesList[2] = filenamesListHigh;

	int c = 0;
/*	for( c=0; c < numCycles; c++){*/
	
		 constructFFANN(&population); // # create initial population
		 
/*
	  for t in range(50): # number of cycles of this evolutionary algorithm
		  #Process the input data through each population member

		  for x in range(popsize): #e.g. for each member FFANN, process it

			#print("lenght of hidden layer is "+str(len(hiddenLayer)))
			print("cycle is "+str(t)+", just about to process member "+str(x))    
			#Run through each line of data in datafile
			process(filenamesList, intendedResult, x) # filename, intendedResult is redundant, number of member, func populates result list

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

