import json
import numpy as np
import math

DATADIR = "data/json/"
chemList = ['URXUHG','URXUCD','URXUPB','LBXBGM','LBXBCD','LBXBPB','LBXTHG','URXUAS']
SEQNDATA = "data/npy/seqn_adult_2007-2014.npy"
CYCLES = ["2007-2008","2009-2010","2011-2012","2013-2014"]

def readJson(fileName):
  """Read and parse JSON from file.
    @param fileName   a file to read
    @return a parsed JSON object.
  """
  with open(fileName, 'r') as inpFile:
    return json.loads(inpFile.read())

labdata0 = readJson(DATADIR+'Laboratory_'+CYCLES[0]+'.json')
labdata1 = readJson(DATADIR+'Laboratory_'+CYCLES[1]+'.json')
labdata2 = readJson(DATADIR+'Laboratory_'+CYCLES[2]+'.json')
labdata3 = readJson(DATADIR+'Laboratory_'+CYCLES[3]+'.json')
labDataList = [labdata0,labdata1,labdata2,labdata3]
seqnList = np.load(SEQNDATA)

out = np.full((seqnList.shape[0],8,2),-1.0)

for i in range(len(seqnList)):

    seqn = str(seqnList[i])

    for lab in labDataList:

        #print(lab)
        #for k in lab:
        #    print(k)

        if seqn in lab:

            #print(seqn,lab[seqn])

            for ch in range(len(chemList)):

                chemName = chemList[ch]
                lcvarName = chemName[0:2] + 'D' + chemName[3:len(chemName)] + 'LC'

                if chemName in lab[seqn]:
                    newVal = lab[seqn][chemName]
                    if not math.isnan(newVal):
                        out[i,ch,0] = newVal

                if lcvarName in lab[seqn]:
                    newVal = lab[seqn][lcvarName]
                    if not math.isnan(newVal):
                        out[i,ch,1] = newVal


np.save('data/npy/labdata_2007-2014.npy',out)