# shapefileAnalysis03a.py

import math
import numpy as np
import cv2
#from math import cos as cos
#from math import sin as sin
from osgeo import ogr
import json
import matplotlib.path as mpltPath
import pickle

from tqdm import tqdm

# numpy versions of trig funcs are vectorized 
# so you can use them on all elements of an array simultaneously
sin = np.sin
cos = np.cos


def pixelCenters(px, py, shiftx, shifty, theta, plotXmax, plotYmax):
    """
    Inputs the parameters of a raster mesh, and returns a list of pixels 
    within that mesh, and within the [0,plotXmax] and [0,plotYmax] extent.
    
    thanasi - rewrite seems to speed up by ~100x on 100x100 grid
    """
    
    hypot = math.sqrt(plotXmax**2 + plotYmax**2)

    bbH = plotYmax # bounding box height
    bbW = plotXmax # bounding box width
    
    # generate oversize grid of centers (qx,qy)
    qx, qy = np.mgrid[-hypot-px/2 : hypot+px/2 : px, 
                      -hypot-py/2 : hypot+py/2 : py]
    
    # make into 1d arrays and shift as desired
    qx = qx.ravel() + shiftx
    qy = qy.ravel() + shifty
    
    # make rotation matrix
    rotmat = np.asarray([[cos(theta), -sin(theta)],
                         [sin(theta), cos(theta)]])
    
    # and stack centers into a 2xN matrix
    qxy = np.vstack([qx[np.newaxis, :], qy[np.newaxis,:]])
    
    # rotate centers 
    qxy2 = rotmat @ qxy
    
    # extract rotated centers
    xc = qxy2[0]
    yc = qxy2[1]
    
    # keep only points that fall strictly within the plot
    m = (xc > 0) & (xc < bbW) & (yc > 0) & (yc < bbH)
    
    # return Nx2 array with only selected points
    return  qxy2.T[m,:]


def pixelCorners(pxlCntrs, px, py, theta):
    """
    Create new array of 4 pixel corners for each pixel center in list 
    
    """
    numPixels = pxlCntrs.shape[0]
    
    # save constants
    h2 = 0.5*math.sqrt(px**2+py**2) # half-hypotenuse: distance from center to each corner
    pi4 = math.pi/4
    
    # create array to populate with corner points
    pxlCrnrs = np.zeros((numPixels, 4, 2))
    

    # you can assign these values a bit faster using numpy notation
    # you could probably even do one better and use some matrix algebra,
    # but this is a bit more readable and much less work....
    # i got ~780x speedup on this function call for 100x100 grid
    
    # first corner
    pxlCrnrs[:,0,0] = pxlCntrs[:,0] + h2*cos(pi4+theta) # x
    pxlCrnrs[:,0,1] = pxlCntrs[:,1] + h2*sin(pi4+theta) # y
    
    # second corner
    pxlCrnrs[:,1,0] = pxlCntrs[:,0] + h2*cos(pi4-theta) # x
    pxlCrnrs[:,1,1] = pxlCntrs[:,1] - h2*sin(pi4-theta) # y
    
    # third corner 
    pxlCrnrs[:,2,0] = pxlCntrs[:,0] - h2*cos(pi4+theta) # x
    pxlCrnrs[:,2,1] = pxlCntrs[:,1] - h2*sin(pi4+theta) # y
    
    # fourth corner
    pxlCrnrs[:,3,0] = pxlCntrs[:,0] - h2*cos(pi4-theta) # x
    pxlCrnrs[:,3,1] = pxlCntrs[:,1] + h2*sin(pi4-theta) # y


    return pxlCrnrs
    
    

def format_func(value, tick_number):
    # find number of multiples of pi/8
    N = int(np.round(value * 8 / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/8$"
    elif N == 2:
        return r"$\pi/4$"
    elif N == 3:
        return r"$3\pi/8$"
    elif N == 4:
        return r"$\pi/2$"
    elif N == 5:
        return r"$5\pi/8$"
    elif N == 6:
        return r"$3\pi/4$"
    elif N == 7:
        return r"$7\pi/8$"
    elif N % 8 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)
        
        
        

   
        
        
def simulateFID(FID, shape, numPlots, pxRange, thetaRange, shiftSteps, rsltPath):
    
    numShifts = shiftSteps**2
        
    print('FID = '+str(FID) + ' of ' + str(numPlots))
    
    plotF = shape.GetFeature(FID)
    plotJ = json.loads(plotF.ExportToJson())
    plotArr = np.array(plotJ['geometry']['coordinates'][0])
    
    # shift plot to origin
    #  AGA:: how is origin defined here?
    plotArr = plotArr - plotArr.min(0)
    
    # define bounding box
    plotDims = plotArr.max(0)
    
    # create blank image
    plotRaster = np.zeros((math.ceil(plotDims[1]), math.ceil(plotDims[0]), 3),
                        dtype=np.uint8)
    pts = np.round(plotArr).reshape((-1,1,2)).astype(np.int32)
    pts2 = ((0,math.ceil(plotDims[1]))-pts)*(-1,1) # flip vertical component so plot shape is upright
    # cv2.polylines(plotRaster, [pts2], True, (255,255,0), 2)
    # cv2.imshow('plotRaster', plotRaster)
    
    ## collect plot statistics
    
    plotGeom = plotF.GetGeometryRef()
    plotArea = plotGeom.GetArea()
    
    # angle of minimum width / minimum width
    
    
    # angle of maximum width / maximum width
    
    plotXmax = plotDims[0]
    plotYmax = plotDims[1]
    

    ## iterate over pxRange and thetaRange to calculate landed pixels

    # pre-make data result array
    landedPxArr = np.zeros((len(pxRange), len(thetaRange)), dtype=np.float64)
    landedPixels = []
    plotMinRects = []
    
    for iPx in tqdm(range(len(pxRange)), desc='iPx loop'):
        px = pxRange[iPx]
        py = px
        for iTheta in tqdm(range(len(thetaRange)), desc='iPx loop', leave=False):
            theta = thetaRange[iTheta]
            print('px = ' + str(px) + ', theta = ' + str(theta))
            
            landedPxTot = 0
            landedPxAvg = 0
            for shiftx in tqdm(np.arange(0, px, px/shiftSteps).tolist(), desc='shiftx loop', leave=False):
                for shifty in np.arange(0,py, py/shiftSteps):
                                    
                    pxlCntrs = pixelCenters(px, py, 
                                            shiftx, shifty, theta, 
                                            plotXmax, plotYmax)
                    pxlCntrs = np.array(pxlCntrs)
                    
                    # # show pxlCntrs placement
                    # sf = 10
                    # canvasCtrs = np.zeros((sf*math.ceil(plotYmax), sf*math.ceil(plotXmax), 3))
                    # for pt in pxlCntrs:
                    #     cv2.drawMarker(canvasCtrs, (int(round(sf*pt[0])), int(round(sf*(plotYmax-pt[1])))), (255,255,0), cv2.MARKER_CROSS, 2)
                    # cv2.imshow('canvasCtrs', canvasCtrs)
                    
                    # find pixel corners based on pixel centers
                    pxlCrnrs = pixelCorners(pxlCntrs, px, py, theta)
                    
                    
                    ## test landing of pixels
                    
                    plotPath = mpltPath.Path(plotArr)
                    pxlCrnrs1col = pxlCrnrs.reshape(-1,2) # reshape array for contains_points fn
                    
                    pxlsLanded = plotPath.contains_points(pxlCrnrs1col, radius=1e-9)
                    
                    pxlsLanded4 = pxlsLanded.reshape(-1,4) # reshape result back to 4-corners-per-row
                    
                    landedPx = 0
                    for pxl in pxlsLanded4:
                        if pxl.all():
                            landedPx += 1

                    landedPxTot += landedPx
                    
            landedPxAvg = landedPxTot / numShifts
            
            landedPxArr[iPx, iTheta] = landedPxAvg
    landedPixels.append(landedPxArr)


    ## establish angle of minimum width (alpha) for that particular plot
    
    plotMinRect = cv2.minAreaRect(plotArr.astype(np.float32))
    plotMinRects.append(plotMinRect)


    ## store data files

    pickle.dump(landedPixels, open(rsltPath+'landedPixels_'+
                                   str(FID).zfill(4)+'.p', 'wb'))
    pickle.dump(plotMinRects, open(rsltPath+'plotMinRects_'+
                                   str(FID).zfill(4)+'.p', 'wb'))
     
    
    ## display data
    
#    import matplotlib.pyplot as plt
#    from mpl_toolkits.mplot3d import Axes3D
#    from matplotlib import cm
#    import matplotlib.ticker as tck
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    
#    # X = pxRange/plotSize
#    # Y = thetaRange
#    Y, X = np.meshgrid(thetaRange, pxRange/plotSize)
#    Z = landedPxArr
#    
#    surf = ax.plot_surface(X, Y, Z, 
#                        cmap=cm.coolwarm, linewidth=0, antialiased=False)
#    # Customize the z axis.
#    ax.set_zlim(0.0, 1.0)
#    # ax.zaxis.set_major_locator(LinearLocator(10))
#    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#    
#    # Add a color bar which maps values to colors.
#    fig.colorbar(surf, shrink=0.5, aspect=5)
#    
#    # apply axis labels
#    ax.set_xlabel('pixel size/plot ize [m/m]')
#    ax.set_ylabel('theta [rad]')
#    ax.set_zlabel('pixel area coverage ratio [m^2/m^2]')
#    # ax.set_xlim(-40, 40)
#    # ax.set_ylim(-40, 40)
#    # ax.set_zlim(-100, 100)
#    # plt.xlabel('px/plotSize [m/m]')
#    # plt.ylabel('theta [rad]')
#    
#    ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 8))
#    # ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 8))
#    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
#    
#    plt.savefig(rsltPath+'PixelAreaRatio_plot'+str(FID).zfill(4)+'.png')
#    plt.savefig(rsltPath+'PixelAreaRatio_plot'+str(FID).zfill(4)+'.pdf')
#    plt.show()

    res = (FID,1)
    return (res,landedPixels, plotMinRects)
    # return










