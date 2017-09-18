textfile = open("/home/pieter/projects/textm/assignment2/data/AmericanGazetteer_1762_6pages.txt", "r")
outputfile = open("/home/pieter/projects/textm/assignment2/output.txt", "w")
lines = textfile.readlines()

newfile = []

for line in lines:
    if ". " in line:
        line = line.replace('. ', '.\n')
        newfile.append(line)
    else:
        newfile.append(line)
    #newline = line.split('.')

clean = []

for line in newfile:
    if len(line.strip()) == 0:
        print line
    else:
        line = line.replace('-\n', '')
        line = line.replace(' \n', '')
        clean.append(line)

print len(clean)

for c in clean:
    outputfile.write(c)



textfile.close()
outputfile.close()
