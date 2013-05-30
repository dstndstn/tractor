from general import general
import sys

f = open('offdata.txt','r')

for line in f:
    name,ra,dec = line.split()
    generalRC3(name,ra,dec)
