import multiprocessing
import keyboard
from Test import Slam
import PlotPoints                                                           

var = Slam()

p1 = multiprocessing.Process(target = var.main)
p2 = multiprocessing.Process(target = PlotPoints.main)

p1.start()
p2.start()

while True:
    if keyboard.is_pressed("q"):
        p1.terminate()
        p2.terminate()
        break

