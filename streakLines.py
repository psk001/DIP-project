import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

currVideo = cv.VideoCapture('UCF_CrowdsDataset/2018-6_70.mov')
numFrames = int(currVideo.get(cv.CAP_PROP_FRAME_COUNT))

ret, frame = currVideo.read();
prevGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

while 1:
    ret, frame = currVideo.read()
    nextGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prevGray, nextGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    px = np.arange(0, flow.shape[1], 10)
    py = np.arange(flow.shape[0], -1, -10)
    dx = flow[::10, ::10, 0]
    dy = -flow[::10, ::10, 1]
    
    plt.quiver(px, py, dx, dy)
    plt.axis('off')
    plt.show()
    prevGray = nextGray.copy()

currVideo.release()
cv.destroyAllWindows()


