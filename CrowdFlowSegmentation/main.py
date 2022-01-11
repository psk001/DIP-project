import numpy as np
import cv2 as cv
from CrowdFlowSegmentation import *

stride=20

cap = cv.VideoCapture('UCF_CrowdsDataset/rush_01.mov')

numFrames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
ret, frame = cap.read()
if not ret:
    print('No frames grabbed!\n')

crowdFlow = CrowdFlowSegmentation(frame, numFrames, stride)

while True:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!\n')
        break
    #modified_frame = crowdFlow.to_streak_flow(frame)
    modified_frame = crowdFlow.to_streaklines(frame)
#    modified_frame = crowdFlow.to_streak_flow(modified_frame)
    #modified_frame = crowdFlow.to_similarities(modified_frame)
#    modified_frame = crowdFlow.to_watershed(modified_frame)
    cv.imshow('window', modified_frame)

    k = cv.waitKey(30) & 0xff
    if (k == 27 or k == ord('q')):
        break
    # to save screenshot of frame press 's'
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame)
        cv.imwrite('opticalhsv.png', modified_frame)

cap.release()
cv.destroyAllWindows()
 