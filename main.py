
import sys
import numpy
import cv2 as cv


def main():

    #1.1 load image and show it
    image = cv.imread("frames/000000.jpg")

    cv.imshow("Display window", image)
    k = cv.waitKey(0)
        
    #1.2 print the shape of the array
    imgAsArray = numpy.asarray(image)
    print(f'shape of array: {imgAsArray.shape}')

    #1.3 print the image itself returns bgr values
    print(image)

    #1.4 show a grayscale image
    grayscaleImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY )

    cv.imshow("Display window", grayscaleImage)
    k = cv.waitKey(0)

    #1.5 save the grayscale file as a png
    if k == ord("s"):
        cv.imwrite("grayscaleframe000000.png", grayscaleImage)

    #Step 2
    #2.1 & 2.2 set up a video capture
    formatted_number = f"{0:06d}"
    filename = f'frames/{formatted_number}.jpg'
    cap = cv.VideoCapture(filename)

    grayFrames = []

    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv.imshow('frame', frame)

        #2.3 get avg image
        grayscaleFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY )
        grayFrames.append(grayscaleFrame)

        if cv.waitKey(30) == ord('q'):
            break
    cap.release()

    #2.3/2.4 show the avg image
    avgFrame = numpy.mean(grayFrames,axis=0).astype('uint8')
    cv.imshow("Average Frame", avgFrame)
    #2.5 save the average image
    cv.imwrite("grayAverageFrame.png", avgFrame)
    cv.waitKey(0)


    #Step 3
    #3.1
    partOneImage = cv.imread('grayscaleframe000000.png', cv.IMREAD_GRAYSCALE)
    partTwoImage = cv.imread('grayAverageFrame.png', cv.IMREAD_GRAYSCALE)

    #3.2 compute absolute difference between the images from part I and part II and display the resulting image
    absDiff = cv.absdiff(partOneImage,partTwoImage);
    print(f'Abs Diff: {absDiff}')
    print('Showing abs diff')
    cv.imshow("absDiff", absDiff)
    cv.waitKey(0)

    #Step 3.3 threshold the absolute difference image and manually find a good threshold value
    ret,thresh1 = cv.threshold(absDiff,30,255,cv.THRESH_BINARY)
    cv.imshow("Absolute Difference Threshold", thresh1)
    cv.waitKey(0)

    #Step 3.4 threshold the absolute difference image using  Otsu's method which will calculate the threshold itself
    ret, thresh2 = cv.threshold(absDiff,0,255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imshow("Otsu Threshold", thresh2)
    cv.waitKey(0)

    #3.4.1 3.4 but filtered with gaussian before the otsu threshold
    blur = cv.GaussianBlur(absDiff,(5,5),0)
    ret, threshBlur = cv.threshold(blur,0,255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imshow("Otsu with Gaussian Blur", threshBlur)
    cv.waitKey(0)

    cv.destroyAllWindows() 
    
    #BONUS
    filename = f'frames/{formatted_number}.jpg'
    cap = cv.VideoCapture(filename)

    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break


        #Contours do not appear
        grayscaleFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY )
        absDiffFrame = cv.absdiff(grayscaleFrame,partTwoImage)
        blur = cv.GaussianBlur(absDiffFrame,(5,5),0)
        ret, thresh3 = cv.threshold(blur,0,255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        contours, hierarchy = cv.findContours(thresh3, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        thresh4 = cv.drawContours(thresh3, contours, -1, (0,255,0), 3)

        if not ret:
            print('threshold failed')
            break
        cv.imshow("Draw Contours", thresh3)

        if cv.waitKey(30) == ord('q'):
            break
    cap.release()
    

if __name__ == '__main__':
    main()


