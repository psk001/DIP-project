import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

currVideo = cv.VideoCapture('UCF_CrowdsDataset/RF1-12977_70.mov')

if(not currVideo):
    raise Exception ("video not captured\n")
numFrames = int(currVideo.get(cv.CAP_PROP_FRAME_COUNT))

#width = currVideo.get(cv.CAP_PROP_FRAME_WIDTH)

ret, frame = currVideo.read();
prevGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

while (currVideo.isOpened()):
    ret, frame = currVideo.read()
    cv.imshow("input", frame)
    nextGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prevGray, nextGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    #cv.imshow("current flow", flow)

    px = np.arange(0, flow.shape[1], 10)
    py = np.arange(flow.shape[0], -1, -10)
    dx = flow[::10, ::10, 0]
    dy = -flow[::10, ::10, 1]

    #cv.imshow("dense optical flow", flow)
    
    plt.quiver(px, py, dx, dy)
    plt.axis('off')
    plt.show()
    prevGray = nextGray.copy()

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

currVideo.release()
cv.destroyAllWindows()


