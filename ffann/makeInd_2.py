
f = open("test.c","a")

for x in range(200,400):
 f.write("superpopulation->newpopulation["+str(x-200)+"] = &ind"+str(x)+";\n");
f.close()


