import numpy as np
from GenAssignPro import GenAssignPro
from GenAssignPro import GenAssignProLR
from GenAssignPro import DualProb
from GenAssignPro import DualProbSurrogateLR

def preProcessData(fileName):
    with open(fileName, 'r') as f:
        genAssignProbData = f.read()

    genAssignProbData = genAssignProbData.strip().split('\n')
    problemSize = genAssignProbData[0].strip().split(' ')
    nAgent, nJob = int(problemSize[0]), int(problemSize[1])
    capacity = genAssignProbData[-1].strip().split(' ')[0]
    b, A = [int(capacity) for i in range(nAgent)], []
    for i in range(1, len(genAssignProbData)):
        temp = genAssignProbData[i].strip().split(' ')
        temp = [int(x) for x in temp]
        for x in temp:
            A.append(x)
        if len(A) == 2 * nAgent * nJob:
            break
        if len(A) > 2 * nAgent * nJob:
            print("constraints data error")

    A = np.array(A).reshape((2*nAgent, nJob))
    costCoeff = A[0:nAgent,:]
    A = A[nAgent:,:]
    return nAgent, nJob, A, b, costCoeff

nAgent, nJob, A, b, costCoeff = preProcessData('a05200.txt')
genAssignPro = GenAssignPro(nAgent, nJob, A, b, costCoeff)
genAssignPro.addConstrs()
genAssignPro.setObjective()
bestUpBound = genAssignPro.solve()
print("best up bound: ", bestUpBound)

lamd = np.zeros(nJob)
genAssignPro = GenAssignProLR(nAgent, nJob, A, b, costCoeff)
genAssignPro.addConstrs()
genAssignDualPro = DualProb(bestUpBound = bestUpBound, lamd = lamd)
maxIterTimes = 20

for iterTimes in range(maxIterTimes):
    genAssignPro.setObjective(lamd)
    origObjValues = genAssignPro.solve()
    subgradients = genAssignPro.computeSubgradients()

    genAssignDualPro.setSubgradients(subgradients)
    genAssignDualPro.getRelaxedObjValues(origObjValues)
    genAssignDualPro.computeStepSize()
    lamd = genAssignDualPro.updateLamd()
    genAssignDualPro.moniterProcess()

genAssignDualPro.printResults()
genAssignDualPro.printLog(fileName = 'GAPLog.txt')
genAssignPro.getFeasibleByheuristic()