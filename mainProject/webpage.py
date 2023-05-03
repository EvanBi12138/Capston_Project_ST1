from funcset import *



dfO = describeDataset()

dfE = contentInterpretation(dfO)

visualData(dfE)

dfE = findThreshold(dfE)

dfN = reduceAttribute(dfE)

dfNN = outlier(dfN)

checkTarget(dfNN)

checkRelation(dfNN)

dfA, dfG = prepareModel(dfNN)

modelOptimization(dfA, dfG)

