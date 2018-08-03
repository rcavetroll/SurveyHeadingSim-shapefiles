# shapefileAnalysis03a.py
# simulation using manual plot data from ESRI shapefile
# based on shapefileAnalysis01, but now using real plot boundaries and statistics

"""
Choose pixel size px [1,30m]
Rotate [0, pi]
Shift [0,px] in both directions.
Apply mask.
Count statistics of resulting "landed" pixels.
"""

import sys
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
# import pprint
from osgeo import ogr
import json
# import pyshp
import matplotlib.path as mpltPath
import pickle
import concurrent.futures
import multiprocessing

import shapefileAnalysis03a


## SETUP
[fileName, startFID, endFID] = sys.argv
startFID = int(startFID)
endFID = int(endFID)


scriptName = 'shapefileAnalysis03a'
srcPath = '../../plotData/'
rsltPath = '../../results/'+scriptName+'/'
plots = srcPath+'plotBorders01.shp'


## open shapefile with OGR
file = ogr.Open(plots)
shape = file.GetLayer(0)
numPlots = shape.GetFeatureCount()


## define simulation parameters
# define ranges
pxMin = 0.5
pxMax = 20.0
pxStep = 0.5

thetaMin = 0.0
thetaMax = math.pi
thetaStep = 0.05

shiftSteps = 16
numShifts = shiftSteps**2

# store metadata in dictionary
simParams = {'pxMin': pxMin, 'pxMax':pxMax, 'pxStep':pxStep,
            'thetaMin':thetaMin, 'thetaMax':thetaMax, 'thetaStep':thetaStep,
            'shiftSteps': shiftSteps, 'numShifts':numShifts}

with open(rsltPath+'simParams.pkl', 'wb') as f:
    pickle.dump(simParams, f, protocol=pickle.HIGHEST_PROTOCOL)


# establish independent variable ranges
pxRange = np.arange(pxMin, pxMax+pxStep, pxStep)
thetaRange = np.arange(thetaMin, thetaMax, thetaStep)

with open(rsltPath+'ranges.pkl', 'wb') as f:
    pickle.dump([pxRange, thetaRange], f)


## run simulation

# startFID = 0
# endFID = 10

results = []
landedPixels = []
plotMinRects = []


## multiprocessing attempt
# # define a function
# processes = [multiprocessing.Process(target=shapefileAnalysis03a.simulateFID, 
#                                      args=(FID, shape, numPlots, pxRange, 
#                                            thetaRange, shiftSteps, rsltPath)) for FID in range(startFID, endFID)]
# 
# # run processes
# for p in processes:
#     p.start()
# 
# 
# # exit the completed processes
# for p in processes:
#     p.join()
# 

## basic approach

for FID in range(startFID, endFID):

    res, landedPxArr, plotMinRect = shapefileAnalysis03a.simulateFID(FID, shape, numPlots, pxRange, thetaRange, shiftSteps, rsltPath)
    
    results.append(res)
    landedPixels.append(landedPxArr)
    plotMinRects.append(plotMinRect)

    pickle.dump(landedPixels, open(rsltPath+'landedPixels_'+
                                str(startFID).zfill(4)+'-'+
                                str(FID).zfill(4)+'.p', 'wb'))
    pickle.dump(plotMinRects, open(rsltPath+'plotMinRects_'+
                                str(startFID).zfill(4)+'-'+
                                str(FID).zfill(4)+'.p', 'wb'))


## collect bounding box data
bboxData = []

startFID = 0
for FID in range(0, numPlots):
    print('FID = '+str(FID) + ' of ' + str(numPlots))
    
    plotF = shape.GetFeature(FID)
    plotJ = json.loads(plotF.ExportToJson())
    plotArr = np.array(plotJ['geometry']['coordinates'][0])
    
    # shift plot to origin
    plotArr = plotArr - plotArr.min(0)
    
    # define bounding box
    plotDims = plotArr.max(0)
    
    bboxDataRow = []
    rotatedArr = np.zeros(plotArr.shape, dtype=np.float64)
    for theta in thetaRange:
    	# rotate shape, point by point, about origin, by theta
        for i in range(len(plotArr)):
            x1, y1 = plotArr[i]
            
            if x1==0 and y1==0:
                x2,y2 = x1,y1
            else:
                h = math.sqrt(x1**2 + y1**2)
                phi = math.atan(y1/x1)
                x2 = h*math.cos(theta+phi)
                y2 = h*math.sin(theta+phi)
            rotatedArr[i] = (x2, y2)
        rotatedArr = rotatedArr - rotatedArr.min(0)
        
        # store maximum x and y components of rotated plot array
        bboxDataRow.append(rotatedArr.max(0))
    bboxData.append(bboxDataRow)

bboxDataArr = np.array(bboxData)

pickle.dump(bboxDataArr, open(rsltPath+'bboxData.pkl', 'wb'))









