import numpy as np
import cv2

cap = cv2.VideoCapture(0)
img1 = cv2.imread('Capture.png')
ref_img = cv2.imread('contrast.png')

count = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img2 = gray
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    

    if len(matches)>3:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        im_out = cv2.warpPerspective(ref_img, homography, (img2.shape[1],img2.shape[0]))

    im_out = cv2.cvtColor(im_out, cv2.COLOR_BGR2GRAY)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:5],None, flags=2)
    img2 = cv2.resize(img2,(640,480))
    added_image = cv2.addWeighted(img2,0.4,im_out,1,0)


    count = count+1
    cv2.imwrite(f'images/frame{count}.png',gray)
    cv2.imwrite(f'images/homography{count}.png',img3)
    cv2.imwrite(f'images/matches{count}.png',np.hstack((im_out,added_image)))
   



    # Display the resulting frame
    cv2.imshow('frame',gray)
    cv2.imshow('homography',im_out)
    cv2.imshow('matches',img3)
    cv2.imshow('overlay',added_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()