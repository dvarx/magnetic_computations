from init_constants import plotFileField
from calibration_postprocessing import kernelAveraging
import numpy as np
from numpy.linalg import pinv,norm
import matplotlib.pyplot as plt
from init_constants import plotField

"""
-This script should be used in conjunction with calib_definitions.py

-This script computes actuation matrices from calibration files. The files should contain N*N*N values,
the order of the position vectors should not play a role

-The end of this script contains code to compute the currents required to
generate a field B and a (projected) gradient gradB. The code also generates a
quiver plot to show the resulting field distribution in the magnetic work space

Inputs:    
    calibrationFileString       string describing the format of the names of the calibration files
    limits                      the limits of the magnetic working space ((xmin,xmax),(ymin,ymax),(zmin,zmax)) [mm]
    plotExampleWorkspace        if this is set to True, the code at the end of the script which plots the field in the entire workspace
                                for a certain desired (field,gradient) combination is executed
"""

calibrationFileString="./calibration/minimag_calibration/mfg-100_00_meas_vfield_0%i.txt"
calibMatrices=[]
limits=[]
plotExampleWorkspace=True
useGradientForPlot=False
smoothingField=False
smoothingGradient=False

for i in range(0,8):
    (mat,dims,limits)=plotFileField(filename=calibrationFileString%(i),loadFlag=True,storeLoc=True)
    calibMatrices.append(mat)

dims=[int(x) for x in dims]
deltas=[(limits[i][1]-limits[i][0])/(dims[i]-1) for i in range(0,3)]
N=dims[0]*dims[1]*dims[2]

#some helper functions to convert integer indices and real coordinates
toIndex=lambda r:[int(round((r[i]-limits[i][0])/deltas[i])) for i in range(0,3)]
toLowestIndex=lambda r:[int((r[i]-limits[i][0])/deltas[i]) for i in range(0,3)]
toPosition=lambda index:[limits[i][0]+index[i]*deltas[i] for i in range(0,3)]
toLinearIndex=lambda cubeindex:cubeindex[0]*dims[2]*dims[1]+cubeindex[1]*dims[2]+cubeindex[2]
    
#index usage: shapeCube[coiln][i][j][k]
shapeCubesRaw=[np.zeros((dims[0],dims[1],dims[2],3)) for i in range(0,8)]

for cuben in range(0,8):
    for i in range(0,N):
        calibMatrix=calibMatrices[cuben]
        index=toIndex(calibMatrix[i][3:6])
        #print("location "+str(calibMatrix[i][3:6]))
        #print("index "+str(index))
        shapeCubesRaw[cuben][index[0]][index[1]][index[2]][:]=calibMatrix[i][0:3]

shapeCubes=kernelAveraging(shapeCubesRaw) if smoothingField else shapeCubesRaw 
   
#compute the derivative actuation matrices dB/dx,dB/dy,dB/dz
#units T/m
#indices of shapeCubesDer: shapeCubesDer[dervn][coiln][i][j][k]
shapeCubesDerRaw=[[np.zeros((dims[0],dims[1],dims[2],3)) for coiln in range(0,8)] for dervn in range(0,3)]

def computeDerCubes(shapeCubes):
    global dims
    #compute the derivative actuation matrices dB/dx,dB/dy,dB/dz
    #units T/m
    #indices of shapeCubesDer: shapeCubesDer[dervn][coiln][i][j][k]
    shapeCubesDerRaw=[[np.zeros((dims[0],dims[1],dims[2],3)) for coiln in range(0,8)] for dervn in range(0,3)]
    for coiln in range(0,8):
        for dervn in range(0,3):
            deltaIndex=np.zeros(3)
            deltaIndex[dervn]=1
            for i in range(0,dims[0]):
                for j in range(0,dims[1]):
                    for k in range(0,dims[2]):
                        index=np.array([i,j,k])
                        indexp=index+deltaIndex
                        indexm=index-deltaIndex
                        index=tuple(index.astype(int))
                        indexp=tuple(indexp.astype(int))
                        indexm=tuple(indexm.astype(int))
                        #if on a face of the cube, use the one-sided difference quotient
                        if(index[dervn]==0):
                            shapeCubesDerRaw[dervn][coiln][index]=(shapeCubes[coiln][indexp]-shapeCubes[coiln][index])/deltas[dervn]
                        elif (index[dervn]==dims[dervn]-1):
                            shapeCubesDerRaw[dervn][coiln][index]=(shapeCubes[coiln][index]-shapeCubes[coiln][indexm])/deltas[dervn]
                        else:
                            shapeCubesDerRaw[dervn][coiln][index]=(shapeCubes[coiln][indexp]-shapeCubes[coiln][indexm])/(2*deltas[dervn])
    return shapeCubesDerRaw
          
#compute the derivative cubes of the magnetic fields                  
shapeCubesDerRaw=computeDerCubes(shapeCubes)

if not smoothingGradient:
    shapeCubesDer=shapeCubesDerRaw
else:  
    shapeCubesDer=[] 
    for dervn in range(0,3):
        shapeCubesDer.append(kernelAveraging(shapeCubesDerRaw[dervn]))

#use this function to get the interpolated actuation matrix A in R^(3,8) at position r from a list of shapefield cubes     
def interpolateCubes(CubeList,r):
    #check if r is within limits, if not raise error
    limitError=False
    for i in range(0,3):
        limitError = limitError or (r[i]<limits[i][0])
        limitError = limitError or (r[i]>limits[i][1])
    if limitError:
        raise Exception("PositionOutOfBounds Error: The positional vector r passed to the Interpolation routine is out of the calibration space")
    A=np.zeros((3,8))
    lowestIndex=toLowestIndex(r)
    x=lowestIndex[0]
    y=lowestIndex[1]
    z=lowestIndex[2]
    #compute the indices of the neighbors
    indices=[(x,y,z),(x+1,y,z),(x+1,y+1,z),(x,y+1,z),(x,y,z+1),(x+1,y,z+1),(x+1,y+1,z+1),(x,y+1,z+1)]
    for i in range(0,8):
        relX=(r[0]-(limits[0][0]+deltas[0]*x))/deltas[0]
        relY=(r[1]-(limits[1][0]+deltas[1]*y))/deltas[1]
        relZ=(r[2]-(limits[2][0]+deltas[2]*z))/deltas[2]
        #compute the trilinear interpolation
        c03=(1-relY)*CubeList[i][indices[0]]+relY*CubeList[i][indices[3]]
        c12=(1-relY)*CubeList[i][indices[1]]+relY*CubeList[i][indices[2]]
        c56=(1-relY)*CubeList[i][indices[5]]+relY*CubeList[i][indices[6]]
        c47=(1-relY)*CubeList[i][indices[4]]+relY*CubeList[i][indices[7]]
        c0312=(1-relX)*c03+relX*c12
        c5647=(1-relX)*c47+relX*c56
        A[:,i]=(1-relZ)*c0312+relZ*c5647
    return A
    
def getB(r):
    return interpolateCubes(shapeCubes,r)
    
def getBx(r):
    return interpolateCubes(shapeCubesDer[0],r)
    
def getBy(r):
    return interpolateCubes(shapeCubesDer[1],r)
    
def getBz(r):
    return interpolateCubes(shapeCubesDer[2],r)

#computes the actuation matrix at position r associated with the magnetiztation m
def getA(r,m):
    if(not type(m)==np.ndarray and m!=None):
        m=np.array(m)
    matB=getB(r)
    Bx=getBx(r)
    By=getBy(r)
    Bz=getBz(r)
    if type(m)==type(None):
        A=np.concatenate((matB,Bx,By,Bz),axis=0)
    else:
        matBx=m.dot(Bx).reshape(1,8)
        matBy=m.dot(By).reshape(1,8)
        matBz=m.dot(Bz).reshape(1,8)
        A=np.concatenate((matB,matBx,matBy,matBz),axis=0)
    return A

def getAalt(r,m):
    delta=1e-6
    m=np.array(m)
    r=np.array(r)
    matB=getB(r)
    Bx=(getB(r+np.array([delta,0,0]))-getB(r+np.array([-delta,0,0])))/(2*delta)
    By=(getB(r+np.array([0,delta,0]))-getB(r+np.array([0,-delta,0])))/(2*delta)
    Bz=(getB(r+np.array([0,0,delta]))-getB(r+np.array([0,0,-delta])))/(2*delta)
    if m.size<3:
        A=np.concatenate((matB,Bx,By,Bz),axis=0)
    else:
        A=np.concatenate((matB,m.dot(Bx).reshape(1,8),m.dot(By).reshape(1,8),m.dot(Bz).reshape(1,8)),axis=0)
    return A


"""
Plot field in workspace for a certain desired (field,gradient) combination
"""
if plotExampleWorkspace:
    desB=np.array([100e-3,0,0])
    if(norm(desB)<1e-4):
        print("Magnetic Flux Density B must not be zero!")
        quit(-1)
    gradB=np.array([0,0,-0.5])
    des=np.concatenate((desB,gradB),axis=0)
    r=(0,0,0)
    A=getA(r,tuple(desB/norm(desB)))
    B=getB(r)
    
    if useGradientForPlot:
        currents=pinv(A).dot(des)
    else:
        currents=pinv(B).dot(desB)
    print("Calibration Files used: %si"%(calibrationFileString[0:-2]))
    print("Position: (%f,%f,%f)"%(r))
    print("Currents required to generate B="+str(desB)+" gradB= "+str(gradB)+":\n"+str(currents))
    if useGradientForPlot:
        print("Actuation Matrix:\n%s"%(str(A)))
    else:
        print("Actuation Matrix:\n%s"%(str(B)))
    
    #make sure that the data frames stored in the calibMatrices list have the same order of positions
    #e.g. the entry with the smallest position [xmin,ymin,zmin] comes first and then [xmin,ymin,zmin+1] etc.
    for coiln in range(0,8):
        calibMatrixNew=np.zeros(calibMatrices[coiln].shape)
        for entryn in range(0,N):
            linearindex=toLinearIndex(toIndex(calibMatrices[coiln][entryn,3:6]))
            calibMatrixNew[linearindex,:]=calibMatrices[coiln][entryn,:]
        calibMatrices[coiln]=calibMatrixNew
    
    B_res=np.zeros((calibMatrices[0].shape[0],3))
    for i in range(0,8):
        for j in range(0,3):
            B_res[:,j]+=calibMatrices[i][:,j]*currents[i]
    
    #fig=plt.figure()
    #axis = fig.add_subplot(111,projection='3d')  
    x,y,z=calibMatrices[0][:,3],calibMatrices[0][:,4],calibMatrices[0][:,5]   
    plotField(x,y,z,B_res[:,0],B_res[:,1],B_res[:,2],scaleFactor=5e-3,ax=None)
    a=input("Type Enter to exit...")