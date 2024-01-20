import cv2
from cvzone.HandTrackingModule import HandDetector
import os
from pathlib import Path  # Added import for pathlib

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread("Resources/Background.png")

# importing all the imgs to a list
folderPathModes = "Resources/Modes"
listImgModes = os.listdir(folderPathModes)
imgModesList = [cv2.imread(os.path.join(folderPathModes, imgModes)) for imgModes in listImgModes]
print(imgModesList)

# importing all the icons to a list
folderPathIcons = "Resources/Icons"
listImgIcons = os.listdir(folderPathIcons)
imgIconsList = [cv2.imread(os.path.join(folderPathIcons, imgIcons)) for imgIcons in listImgIcons]
print(imgIconsList)

modeType = 0  # for changing selection mode
selection = -1
counter = 0
selectionSpeed = 15

detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)
modePositions = [(1136, 196), (1000, 384), (1136, 581)]
counterPause = 0
selectionList = [-1, -1, -1]
while True:
    success, img = cap.read()
    # find hand and its landmark
    hands, img = detector.findHands(img, draw=True, flipType=True)

    # overlaying the webcam feed on the background image
    imgBackground[139:139 + 480, 50:50 + 640] = img
    imgBackground[0:720, 847:1280] = imgModesList[modeType]

    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands and counterPause == 0 and modeType < 3:
        # Information for the first hand detected
        hand1 = hands[0]  # Get the first hand detected
        fingers1 = detector.fingersUp(hand1)
        print(fingers1)

        if fingers1 == [0, 1, 0, 0, 0]:
            if selection != 1:
                counter = 1
            selection = 1

        elif fingers1 == [0, 1, 1, 0, 0]:
            if selection != 2:
                counter = 1
            selection = 2

        elif fingers1 == [0, 1, 1, 1, 0]:
            if selection != 3:
                counter = 1
            selection = 3

        else:
            selection = -1
            counter = 0

        if counter > 0:
            counter += 1
            print(counter)

            cv2.ellipse(imgBackground, modePositions[selection-1], (103, 103), 0, 0,
                        counter*selectionSpeed, (0, 255, 0), 20)
            if counter*selectionSpeed > 360:
                selectionList[modeType] = selection
                modeType += 1
                counter = 0
                selection = -1
                counterPause = 1

    # to pause after each selection is made
    if counterPause > 0:
        counterPause += 1
        if counterPause > 60:
            counterPause = 0

    # Add selection icon at bottom
    if selectionList[0] != -1:
        imgBackground[636:636+65, 133:133+65] = imgIconsList[selectionList[0]-1]
    if selectionList[1] != -1:
        imgBackground[636:636+65, 340:340+65] = imgIconsList[2+selectionList[1]]
    if selectionList[2] != -1:
        imgBackground[636:636+65, 542:542+65] = imgIconsList[5+selectionList[2]]

    # displaying img
    # cv2.imshow("Image", img)
    cv2.imshow("Background", imgBackground)
    cv2.waitKey(1)
