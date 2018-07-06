import json
import pprint
import ast
def parseDict(filePart):
    return {}
def parseFilePart(filePart):
    dict_list = {}
    for i in xrange(0,len(filePart) - 1,2):
        if(isinstance(filePart[i+1], list)):
            if(isinstance(filePart[i+1][0], list)):
                dict_list[filePart[i]] = parseFilePart(filePart[i+1])
                # print "went recursion"
            elif(isinstance(filePart[i+1][0], unicode)):
                print "|{}| ==== |{}|".format(filePart[i] , filePart[i+1][0])
                if(filePart[i+1][0][0] == '['):
                    print "hmm"
                dict_list[filePart[i]] = parseFilePart(filePart[i+1])
        elif(type(filePart[i+1]) == type({})):
            # print "dict" , filePart[i+1]
            dict_list[filePart[i]] = filePart[i+1]
        else:
            dict_list[filePart[i]] = filePart[i+1]
    return dict_list



filename = "../change.geo"
f = open(filename)
fileLol =  json.load(f)
f.close()
dict_list = parseFilePart(fileLol)

    




