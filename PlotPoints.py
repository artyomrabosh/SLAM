import OpenGL.GL as gl
import pangolin
import numpy as np
from Extract import data 
import threading
from Extract import Slam
import time
import cv2

class display3D:
    def __init__(self, newCoordinates):
        self.Coordinates = newCoordinates
        self.viewer()

    def viewer(self):
        pangolin.CreateWindowAndBind('Plot', 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Define Projection and initial ModelView matrix
        self.scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200),
        pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))

        handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
        self.dcam.SetHandler(handler)

        self.dcam.Activate()
        
    def refresh(self):
        if True:
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(0.0, 0.0, 0.0, 1.0)            
            self.dcam.Activate(self.scam)

            gl.glPointSize(5)
            gl.glColor3f(1.0, 0.0, 0.0)
            # # access numpy array directly(without copying data), array should be contiguous.
            data.data = data.data.squeeze()
            pangolin.DrawPoints(data.data)
            pangolin.FinishFrame()


try:
    slam = Slam()
    od = display3D(data.data)

    slam.updateCoordinates()
    od.dcam.Activate(od.scam)
    while not pangolin.ShouldQuit() and not (cv2.waitKey(1) & 0xFF == ord('q')):
        if slam.vid1.isOpened() and slam.vid2.isOpened():
            slam.camera()
        od.refresh()

except Exception as e:
    print(e)
    raise
