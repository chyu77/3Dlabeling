# Standard imports
import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt

square_size = 0.22     
pattern_size = (11, 11) 

pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size
objpoints = []
imgpoints = []

def find_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    params = cv2.SimpleBlobDetector_Params()
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 100000

    params.minDistBetweenBlobs = 100

    params.filterByColor = True
    params.filterByConvexity = True

    # tweak these as you see fit
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.8

    # # # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.15

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.15

    params.minThreshold = 0.45

    blobDetector = cv2.SimpleBlobDetector_create(params)
    ret, corners = cv2.findCirclesGrid(gray, pattern_size, cv2.CALIB_CB_SYMMETRIC_GRID, blobDetector, None)
    if ret:
        cv2.cornerSubPix(gray, corners, pattern_size, (-1, -1), criteria)
        return ret, corners
    return ret, None

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - img2上の点に対応するエピポーラ線を描画する画像
        lines - 対応するエピポーラ線 '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

images = glob.glob('CamCalib\*.bmp')

for fname in images:
    print("FindImg" + fname)
    # Read image
    img= cv2.imread(fname)

    ret, corners = find_corners(img)
    if ret:
        imgpoints.append(corners.reshape(-1, 2))
        objpoints.append(pattern_points)

print("calculating camera parameter...")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# mtx, dist: init para.
np.save("mtx", mtx) 
np.save("dist", dist.ravel()) 

print("RMS = ", ret)
print("mtx = \n", mtx)
print("dist = ", dist.ravel())

K = mtx
dist_coef = dist

# undistort Images.
img_L= cv2.imread('StereoCalib\left0.bmp')
img_R= cv2.imread('StereoCalib\\right0.bmp')

img1 = cv2.undistort(img_L, K, dist_coef)
img2 = cv2.undistort(img_R, K, dist_coef)

# ORB (Oriented FAST and Rotated BRIEF)
detector = cv2.ORB_create()
#detector = cv2.AKAZE_create()

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

kp1,des1 = detector.detectAndCompute(img1,None)
kp2,des2 = detector.detectAndCompute(img2,None)

# Match descriptors.
matches = bf.match(des1,des2)

good = []
pts1 = []
pts2 = []

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

count = 0
for m in matches:
    count+=1
    if count < 60:
        good.append([m])
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)   
        
pts1 = np.float32(pts1)
pts2 = np.float32(pts2)
#E, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)

pts1_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), cameraMatrix=K, distCoeffs=None)
pts2_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), cameraMatrix=K, distCoeffs=None)

E, mask = cv2.findEssentialMat(pts1_norm, pts2_norm, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=3.0)
points, R, t, mask = cv2.recoverPose(E, pts1_norm, pts2_norm)

M_r = np.hstack((R, t))
M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

P_l = np.dot(K,  M_l)
P_r = np.dot(K,  M_r)
point_4d_hom = cv2.triangulatePoints(P_l, P_r, np.expand_dims(pts1, axis=1), np.expand_dims(pts2, axis=1))
point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
point_4d = point_4d[:3, :].T

Kinv = np.linalg.inv(K)
Kinvt = np.transpose(Kinv)
F = np.dot(Kinvt,E,K)

print(F)
