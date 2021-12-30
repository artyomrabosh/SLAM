import cv2
import numpy as np

class Holder:
  def __init__(self, value):
    self._data = value
  @property
  def data(self):
    return self._data
  @data.setter
  def data(self, value):
    self._data = value

data = Holder(None)

class Slam:
    def __init__(self):
        self.vid1 = cv2.VideoCapture(0)
        self.vid2 = cv2.VideoCapture(2)
        self.frame1 = self.vid1.read()[1]
        self.frame2 = self.vid2.read()[1]
        self.list_kp1 = []
        self.list_kp2 = []
        self.cam1 = []
        self.cam2 = []
        self.img1 = None
        self.img2 = None
        self.pts1 = []
        self.pts2 = []
        self.Fundamental = None
        self.mask = None
        self.cameraMatrix = np.array([[164.94015221, 0, 379.98164181], [0, 165.18602708, 353.10553008], [0, 0, 1.0]])
        self.F = None
        self.E = None
        self.frameCount = 0
        self.mask = None
        self.fundamentalMatrix()

    def camera(self):
        self.F, self.mask, self.E, self.pts1, self.pts2 = self.fundamentalMatrix()
        newCoordinates = data.data = self.poseEstimation()
        self.PnP(newCoordinates)

        ret1, frame1 = self.vid1.read()
        ret2, frame2 = self.vid2.read()

        if ret1:
            cv2.imshow('Cam 1', frame1)
            self.frameCount += 30
            self.vid1.set(1, self.frameCount)
        else:
            self.vid1.release()
            return
        if ret2:
            cv2.imshow('Cam 2', frame2)
            self.frameCount += 30
            self.vid2.set(1, self.frameCount)
        else:
            self.vid2.release()
            return

        self.cam1.append(frame1)
        self.cam2.append(frame2)

    def fundamentalMatrix(self):
        try:
            self.pts1 = []
            self.pts2 = []
            self.img1 = cv2.cvtColor(self.frame1, cv2.IMREAD_GRAYSCALE)
            self.img2 = cv2.cvtColor(self.frame2, cv2.IMREAD_GRAYSCALE)

            orb = cv2.ORB_create(nfeatures=1000000, scoreType=cv2.ORB_FAST_SCORE)

            kp1, des1 = orb.detectAndCompute(self.img1, None)
            kp2, des2 = orb.detectAndCompute(self.img2, None)

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            for mat in matches:
                # Get the matching key points for each of the images
                img1_idx = mat.queryIdx
                img2_idx = mat.trainIdx

                # x - columns
                # y - rows
                # Get the coordinates
                (x1, y1) = kp1[img1_idx].pt
                (x2, y2) = kp2[img2_idx].pt

                # Append to each list
                self.list_kp1.append((x1, y1))
                self.list_kp2.append((x2, y2))

            for m in matches:
                if m.distance < 30:
                    self.pts2.append(kp2[m.trainIdx].pt)
                    self.pts1.append(kp1[m.queryIdx].pt)

            self.pts1 = np.float32(self.pts1)
            self.pts2 = np.float32(self.pts2)

            # E is Essential matrix, F is Fundamental matrix
            # Use 8 point algorithm to find fundamental matrix
            self.F, mask = cv2.findFundamentalMat(self.pts1, self.pts2, method=cv2.FM_8POINT, confidence=0.999)
            self.E, mask = cv2.findEssentialMat(self.pts1, self.pts2, self.cameraMatrix, method=cv2.LMEDS, prob=0.999,
                                                threshold=3.0)

            # print("This is the Fundamental Matrix")
            # print(self.F)
            # print("This is the Essential Matrix")
            # print(self.E)

            self.pts1 = self.pts1[mask.ravel() == 1]
            self.pts2 = self.pts2[mask.ravel() == 1]

            if self.F is None or self.F.shape == (1, 1):
                # no fundamental matrix found
                raise Exception('No fundamental matrix found')
            elif self.F.shape[0] > 3:
                # more than one matrix found, just pick the first
                self.F = self.F[0:3, 0:3]
            self.Fundamental = np.matrix(self.F)
            self.mask = mask

            return self.F, self.mask, self.E, self.pts1, self.pts2

        except Exception as e:
            print(e)
            raise

    def poseEstimation(self):
        # SVD Decompostion to find U, S, and V
        # V can be transposed, watch out
        # Remove outlier points later.
        row_width = 4

        retval, R, t, mask = cv2.recoverPose(self.E, self.pts1, self.pts2, self.cameraMatrix)
        extrinsic = np.vstack((np.hstack((R, t.reshape(-1, 1))), [0, 0, 0, 1]))

        cameraMatrix = np.hstack([self.cameraMatrix, np.full((3, 1), 0.0)])

        intrinsic = np.vstack([cameraMatrix, np.full((1, row_width), 0.0)])
        intrinsic[3, 3] = 1.0

        projectionMatrix = np.dot(intrinsic, extrinsic)

        Z = np.delete(projectionMatrix, (-1), axis=0)

        self.pts1 = np.transpose(self.pts1)
        self.pts2 = np.transpose(self.pts2)

        triangulate = cv2.triangulatePoints(Z, np.delete(intrinsic, (-1), axis=0), self.pts1, self.pts2)

        last_column = triangulate[3]

        coordinates1 = np.divide(triangulate, last_column)
        newCoordinates = np.delete(coordinates1, 3, axis=0)


        data.data = np.delete(coordinates1, 3,  axis=0) 

        return newCoordinates

    def PnP(self, newCoordinates):
        distCoeffs = np.zeros((5, 1))

        pts1 = np.array(self.pts1)

        imagePoints1 = pts1.transpose()[:,None,:]
        newCoordinates = newCoordinates.transpose()[:,None,:]

        data.data = newCoordinates

        retval, rvec, tvec = cv2.solvePnP(data.data, imagePoints1, self.cameraMatrix, distCoeffs)

        return data
        
    def updateCoordinates(self):
        newCoordinates = self.poseEstimation()
        data.data = newCoordinates
        self.PnP(newCoordinates)

# od = Slam()
# od.updateCoordinates()

