from twogalaxies import twogalaxies
import sys
f = open('2data.txt','r')

for line in f:
    name1,ra1,dec1,name2,ra2,dec2 = line.split()
    twogalaxies(name1,ra1,dec1,name2,ra2,dec2)

