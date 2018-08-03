# shapefileAnalysis03a.py

import math
import numpy as np
import cv2
from math import cos as cos
from math import sin as sin
from osgeo import ogr
import json
import matplotlib.path as mpltPath
import pickle



def pixelCenters(px, py, shiftx, shifty, theta, plotXmax, plotYmax):
    """
    Inputs the parameters of a raster mesh, and returns a list of pixels 
    within that mesh, and within the [0,plotXmax] and [0,plotYmax] extent.
    """
    
    pxlCntrs = []
    hypot = math.sqrt(plotXmax**2 + plotYmax**2)

    bbH = plotYmax # bounding box height
    bbW = plotXmax # bounding box width
    
    # iterate over original oversized array (before rotation)
    # - oversized array contains any point that could be rotated *into* the 
    #   extent window of the main image bounding box.
    # - oversized array: X: [-hypot,hypot], Y: [-hypot, bbH]
    # - extracting pixel *centers* from this 2D range
    # for queryx in np.arange(-px+px/2, hypot+px/2, px):
    for queryx in np.arange(-hypot+px/2, hypot+px/2, px):
        for queryy in reversed(np.arange(-py+py/2, -hypot+py/2, -py)):
            # translate queryPt in linear space
            qx = queryx + shiftx
            qy = queryy + shifty
            
            # rotate about origin
            if qx != 0:
                phi = math.atan(qy/qx)
            else:
                phi = math.pi/2
            h = math.sqrt(qx**2 + qy**2)
            qx = h * math.cos(theta+phi)
            qy = h * math.sin(theta+phi)
            
            if 0 < qx < bbW and 0 < qy < bbH:
                pxlCntrs.append((qx, qy))
        
        for queryy in np.arange(py/2, bbH+py/2, py):
            # translate queryPt in linear space
            qx = queryx + shiftx
            qy = queryy + shifty
            
            # rotate about origin
            if qx != 0:
                phi = math.atan(qy/qx)
            else:
                phi = math.pi/2
            h = math.sqrt(qx**2 + qy**2)
            qx = h * math.cos(theta+phi)
            qy = h * math.sin(theta+phi)
            
            if 0 < qx < bbW and 0 < qy < bbH:
                pxlCntrs.append((qx, qy))

    return pxlCntrs


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
    
    for i in range(numPixels):
        ctr = pxlCntrs[i]
        
        cnr1 = (ctr[0]+h2*cos(pi4+theta), ctr[1]+h2*sin(pi4+theta))
        cnr2 = (ctr[0]+h2*cos(pi4-theta), ctr[1]-h2*sin(pi4-theta))
        cnr3 = (ctr[0]-h2*cos(pi4+theta), ctr[1]-h2*sin(pi4+theta))
        cnr4 = (ctr[0]-h2*cos(pi4-theta), ctr[1]+h2*sin(pi4-theta))

        pxlCrnrs[i] = [cnr1, cnr2, cnr3, cnr4]


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
    plotArr = plotArr - plotArr.min(0)
    
    # define bounding box
    plotDims = plotArr.max(0)
    
    # create blank image (used for testing)
    # plotRaster = np.zeros((math.ceil(plotDims[1]), math.ceil(plotDims[0]), 3),
    #                     dtype=np.uint8)
    # pts = np.round(plotArr).reshape((-1,1,2)).astype(np.int32)
    # pts2 = ((0,math.ceil(plotDims[1]))-pts)*(-1,1) # flip vertical component so plot shape is upright
    # cv2.polylines(plotRaster, [pts2], True, (255,255,0), 2)
    # cv2.imshow('plotRaster', plotRaster)
    
    
    ## collect plot statistics
    plotGeom = plotF.GetGeometryRef()
    plotArea = plotGeom.GetArea()
        
    plotXmax = plotDims[0]
    plotYmax = plotDims[1]
    

    ## iterate over pxRange and thetaRange to calculate landed pixels
    # pre-make data result array
    landedPxArr = np.zeros((len(pxRange), len(thetaRange)), dtype=np.float64)
    landedPixels = []
    plotMinRects = []
    
    for iPx in range(len(pxRange)):
        px = pxRange[iPx]
        py = px
        for iTheta in range(len(thetaRange)):
            theta = thetaRange[iTheta]
            print('px = ' + str(px) + ', theta = ' + str(theta))
            
            landedPxTot = 0
            landedPxAvg = 0
            for shiftx in np.arange(0, px, px/shiftSteps):
                for shifty in np.arange(0,py, py/shiftSteps):
                                    
                    pxlCntrs = pixelCenters(px, py, 
                                            shiftx, shifty, theta, 
                                            plotXmax, plotYmax)
                    pxlCntrs = np.array(pxlCntrs)
                    
                    # # show pxlCntrs placement (for testing)
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










