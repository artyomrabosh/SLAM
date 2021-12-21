import OpenGL.GL as gl
import pangolin

import numpy as np

class display3D():

    def __init__(self):
        #self.Coordinates = newCoordinates
        #pass in newCoordinates from other class
        pass

    def viewer(self):
        pangolin.CreateWindowAndBind('Main', 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Define Projection and initial ModelView matrix
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 640//2, 480//2, 0.2, 10000),
            pangolin.ModelViewLookAt(0, -10, -8,
                                0, 0, 0,
                                0, -1, 0))
        handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, 640.0/480.0)
        self.dcam.SetHandler(handler)
        self.dcam.Activate()
        
    def refresh(self):
        while not pangolin.ShouldQuit():
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(0.0, 0.0, 0.0, 1.0)            
            self.dcam.Activate(self.scam)
                        
            # Draw camera
            gl.glColor3f(0.0, 1.0, 0.0)
            #pangolin.DrawCameras(pose)
            #add in pose value in main


            gl.glPointSize(5)
            gl.glColor3f(1.0, 0.0, 0.0)
            #pangolin.DrawPoints(newCoordinates)
            
        pangolin.FinishFrame()

    def main(self):
        self.viewer()
        self.refresh()

od = display3D()
od.main()
