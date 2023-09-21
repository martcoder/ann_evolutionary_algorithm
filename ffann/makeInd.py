
f = open("individual_aux.c","a")

for x in range(400):
 f.write("ind"+str(x)+".inputLayer = &inputNode"+str(x)+";\n")
 f.write("ind"+str(x)+".hiddenLayer = (Node**) malloc(nodeSizeMemory * hiddenMax);\n");
 f.write("ind"+str(x)+".outputLayer = (Node**) malloc(nodeSizeMemory * outputLayerLength);\n");
 for y in range(20):
  f.write("ind"+str(x)+".hiddenLayer["+str(y)+"] = &hiddenNode"+str(x)+"_"+str(y)+";\n")
 for z in range(3):
  f.write("ind"+str(x)+".outputLayer["+str(z)+"] = &outputNode"+str(x)+"_"+str(z)+";\n")
 f.write("\n")
f.close()


