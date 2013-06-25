from general import generalRC3
import sys

f = open('offdata.txt','r')

for line in f:
    name,ra,dec = line.split()
    print name,ra,dec
    generalRC3(name.replace('_',' '),ra=ra,dec=dec)
