import cv2

def detectFace(file):
    dst_name = 'imgs/dst.png'

    # Use cascade classifier
    face_classifier = cv2.CascadeClassifier('pretrained_classifier.xml')
    # Read the image
    img = cv2.imread(file)
    # Turn it into gray-image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect face
    faces = face_classifier.detectMultiScale(gray_img, 1.3, 5)
    # Draw face bounding boxes
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)

    # Show and store picture
    cv2.namedWindow('Face-Detected-Result')
    cv2.imshow('Face-Detected-Result', img)
    cv2.imwrite(dst_name, img)

    cv2.waitKey(0)

    return dst_name

def drawText(dst):
    img = cv2.imread(dst)
    res = cv2.putText(img, 'Lena', (50, 80), cv2.FONT_HERSHEY_COMPLEX, 1.0, (200, 0, 100), 2)

    # Show and store picture
    cv2.namedWindow('Face-Detected-Result')
    cv2.imshow('Face-Detected-Result', res)
    cv2.imwrite(dst, res)

    cv2.waitKey(0)
    
if __name__ == '__main__':
    filename = 'imgs/lena.png'
    dst = detectFace(filename)
    drawText(dst)
