import json
import pprint

filename = "/home/brian/Desktop/fire_test_001.geo"
f = open(filename)
fileLol =  json.load(f)

# pprint.pprint(fileLol[11])
# pprint.pprint(fileLol[13])
# pprint.pprint(fileLol[15])
pprint.pprint(fileLol[17])
# pprint.pprint(fileLol[19])
# pprint.pprint(fileLol[21])
# pprint.pprint(fileLol[23])
# print fileLol["software"]
# print fileLol['info']

