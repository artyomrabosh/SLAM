import OpenGL.GL as gl
import pangolin
import numpy as np
from multiprocessing import Process, Queue
from Test import Slam

def main():
    newSlamInstance = Slam()
    r,t,tri,newCoords = newSlamInstance.poseEstimation() 

    pangolin.CreateWindowAndBind('Main', 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Define Projection and initial ModelView matrix
    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200),
        pangolin.ModelViewLookAt(0, -10, -8,
                                0, 0, 0,
                               0, -1, 0))
    handler = pangolin.Handler3D(scam)

    # Create Interactive View in window
    dcam = pangolin.CreateDisplay()
    dcam.SetHandler(handler)

    trajectory = [[0, -5, 6]]
    for i in range(300):
        trajectory.append(trajectory[-1] + np.random.random(3)-0.5)
    trajectory = np.array(trajectory)

    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        dcam.Activate(scam)
        
        gl.glColor3f(0.0, 1.0, 0.0)
        gl.glPointSize(5)
        points = newSlamInstance.newCoordinate

        pangolin.DrawPoints(points)

    pangolin.FinishFrame()



if __name__ == '__main__':
    main()

