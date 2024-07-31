import os
import cv2
cap=cv2.VideoCapture(0)
directory='Image/'
while True:
    _, frame = cap.read() #capturing single frame
    count = {
        'A': len(os.listdir(directory + "/A")),
        'B': len(os.listdir(directory + "/B")),
        'C': len(os.listdir(directory + "/C")),
        'D': len(os.listdir(directory + "/D")),
        'Hello': len(os.listdir(directory + "/Hello")),
        'I_Love_You': len(os.listdir(directory + "/I_Love_You")),
        'Okay': len(os.listdir(directory + "/Okay")),
        'Telephone': len(os.listdir(directory + "/Telephone")),
        'Why': len(os.listdir(directory + "/Why")),
        'Yes': len(os.listdir(directory + "/Yes")),
        '1': len(os.listdir(directory + "/1")),
        '2': len(os.listdir(directory + "/2")),
        '3': len(os.listdir(directory + "/3")),
        '4': len(os.listdir(directory + "/4"))
    }

    row = frame.shape[1]
    col = frame.shape[0]
    cv2.rectangle(frame,(0,40),(300,400),(255,255,255),2)
    cv2.imshow("data",frame)
    cv2.imshow("ROI",frame[40:400,0:300])
    frame=frame[40:400,0:300]
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(directory + 'A/' + str(count['A']) + '.png', frame)
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(directory + 'B/' + str(count['B']) + '.png', frame)
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(directory + 'C/' + str(count['C']) + '.png', frame)
    if interrupt & 0xFF == ord('d'):
        cv2.imwrite(directory + 'D/' + str(count['D']) + '.png', frame)
    if interrupt & 0xFF == ord('h'):
        cv2.imwrite(directory + 'Hello/' + str(count['Hello']) + '.png', frame)
    if interrupt & 0xFF == ord('i'):
        cv2.imwrite(directory + 'I_Love_You/' + str(count['I_Love_You']) + '.png', frame)
    if interrupt & 0xFF == ord('o'):
        cv2.imwrite(directory + 'Okay/' + str(count['Okay']) + '.png', frame)
    if interrupt & 0xFF == ord('t'):
        cv2.imwrite(directory + 'Telephone/' + str(count['Telephone']) + '.png', frame)    
    if interrupt & 0xFF == ord('w'):
        cv2.imwrite(directory + 'Why/' + str(count['Why']) + '.png', frame)
    if interrupt & 0xFF == ord('y'):
        cv2.imwrite(directory + 'Yes/' + str(count['Yes']) + '.png', frame)
    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(directory + '1/' + str(count['1']) + '.png', frame)
    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(directory + '2/' + str(count['2']) + '.png', frame)
    if interrupt & 0xFF == ord('3'):
        cv2.imwrite(directory + '3/' + str(count['3']) + '.png', frame)
    if interrupt & 0xFF == ord('4'):
        cv2.imwrite(directory + '4/' + str(count['4']) + '.png', frame)

cap.release()
cv2.destroyAllWindows()