import re

#author Pieter Wolfert s4220366

def loadfiles():
    outputfile = open("/home/pieter/projects/textm/assignment2/output.txt", "w")
    textfile = open("/home/pieter/projects/textm/assignment2/data/AmericanGazetteer_1762_6pages.txt", "r")
    lines = textfile.read()
    return lines, outputfile, textfile

def main():
    lines, outputfile, textfile = loadfiles()
    newfile = []
    for cha in lines:
        if cha == '-' :
            newfile.append(' ')
        else:
            newfile.append(cha)
    seconditer = []

    #remove the end /splitted sentences
    for i, cha in enumerate(newfile):
        if i < len(newfile)-1:
            if newfile[i] == '\n' and newfile[i-1] != '\n' and newfile[i+1] != '\n':
                seconditer.append(' ')
            else:
                seconditer.append(cha)
    thirditer = []
    #split text

    for i, cha in enumerate(seconditer):
        if cha == '.' and seconditer[i+1] == ' ':
            thirditer.append(cha)
            thirditer.append('\n')
        else:
            thirditer.append(cha)

    for j in thirditer:
        outputfile.write(j)
    close(textfile, outputfile)

def close(textfile, outputfile):
    textfile.close()
    outputfile.close()

if __name__ == '__main__':
    main()
