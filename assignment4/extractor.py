import re
import datetime
#author Pieter Wolfert s4220366

def loadfiles():
    outputfile = open("/home/pieter/projects/textm/assignment4/output.txt", "w")
    textfile = open("/home/pieter/projects/textm/assignment4/phil.txt", "r") \
    #dr phil file
    #textfile = open("/home/pieter/projects/textm/assignment4/input.txt", "r") \
    #merkel file
    lines = textfile.read()
    return lines, outputfile, textfile

def makeStrings(textfile, outputfile):
    """
    Converting a list of characters to a list of strings.
    """
    textstrings = []
    newstring = ' '
    for cha in textfile:
        if cha != '\n':
            newstring += cha
        else:
            textstrings.append(newstring)
            newstring = ' '
    for i, s in enumerate(textstrings):
        if len(s) > 1:
            if s[0].isspace():
                textstrings[i] = s[1:]
            if s[0].isspace() and s[1].isspace():
                textstrings[i] = s[2:]
    return textstrings

def timeISOC1(timestring):
    """
    To iso encoding, simples option.
    """
    if timestring.isdigit():
        return timestring + "-01-01T00:00:00"

def timeMonthYear(timestring):
    """
    Returns date in isoformat.
    """
    date = datetime.datetime.strptime(timestring, '%B %Y')
    return date.isoformat()

def timeYear(timestring):
    """
    Returns date in isoformat.
    """
    timestring = timestring.replace(",", "")
    date = datetime.datetime.strptime(timestring, '%B %d %Y')
    return date.isoformat()

def monthsConvert(month):
    """
    Converts months to their index in a year.
    """
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',\
        'August', 'September', 'October', 'November', 'December']
    return months.index(month) + 1

def getDate(text):
    """
    Gets the date per sentence.
    """
    datelist = []
    regex1 = r"((January|February|March|April|May|June|July|August|September \
        |October|November|December) (\d{1,2}), (\d{4}))"
    regex2 = r"((January|February|March|April|May|June|July|August|September \
        |October|November|December) (\d{4}))"
    regex3 = r"(\d{4})"
    for s in text:
        match = re.search(regex1, s)
        match2 = re.search(regex2, s)
        match3 = re.search(regex3, s)
        if match:
            datelist.append((timeYear(match.group(0)), s))
        elif match2:
            datelist.append((timeMonthYear(match2.group(0)), s))
        elif match3:
            datelist.append((timeISOC1(match3.group(0)), s))
    return datelist

def textcleaner(lines, outputfile, textfile):
    """
    Main method of the program.
    """
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
            if newfile[i] == '\n' and newfile[i-1] != '\n' \
                and newfile[i+1] != '\n':
                seconditer.append(' ')
            else:
                seconditer.append(cha)
    thirditer = []
    #split text
    for i, cha in enumerate(seconditer):
        if cha == '.' and len(seconditer) > i+1 and seconditer[i+1] == ' ':
            thirditer.append(cha)
            thirditer.append('\n')
        else:
            thirditer.append(cha)
    #for c in thirditer:
    #    outputfile.write(c)
    return thirditer

def printtable(datelist):
    """
    Prints table in nice format.
    """
    datelist = sorted(datelist, key=lambda x: x[0])
    for (i, j) in datelist:
        print '{:<15} {}'.format(i, j)


def close(textfile, outputfile):
    textfile.close()
    outputfile.close()

if __name__ == '__main__':
    lines, outputfile, textfile = loadfiles()
    textblob = textcleaner(lines, outputfile, textfile)
    textStrings = makeStrings(textblob, outputfile)
    datelist = getDate(textStrings)
    printtable(datelist)
    close()
