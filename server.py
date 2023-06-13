import socket
import io 
import struct 
from PIL import Image 
import cv2
import numpy as np 
from glob import glob
import time
import sys 


#socket.AF_INET, socket.SOCK_STREAM
server_socket =  socket.socket() 
# server_socket.connect(("raspberrypi", 8000))
server_socket.bind(('0.0.0.0', 8000))

server_socket.listen(0)

connection = server_socket.accept()[0].makefile('rb')

FPS = 10 

def write_video(file_name, images, slide_time=5):
    fourcc = cv2.VideoWriter.fourcc(*'MJPG')
    out = cv2.VideoWriter(file_name, fourcc, FPS, (870, 580))

    for image in images:
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for _ in range(slide_time * FPS):
            cv_img = cv2.resize(image, (870, 580))
            out.write(cv_img)

    out.release()



try:
    while True:
        # Read the length of the image as a 32-bit unsigned int. If the
        # length is zero, quit the loop
        image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
        if not image_len:
            break
        # Construct a stream to hold the image data and read the image
        # data from the connection
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))
        # Rewind the stream, open it as an image with PIL and do some
        # processing on it
        image_stream.seek(0)
        # pil_image = Image.open(image_stream)
        data = image_stream.read()
        # print(data)
        img_array = np.frombuffer((data), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # cv2.imshow("test", im)
        # cv2.waitKey(1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # cv2.imshow("test", hsv)
            # input color 
        def num_to_str(color):
            switcher = {
                "red" :  np.array([[[0,0,255]]], dtype=np.uint8),
                "blue" :  np.array([[[255,0,0]]], dtype=np.uint8),
                "green" :  np.array([[[0,255,0]]], dtype=np.uint8),
                "yellow" :  np.array([[[0,255,255]]], dtype=np.uint8),
            }
            return switcher.get(color, "nothing")
        
        stage1_color_input = sys.argv[1]
        stage1_color_hsv = cv2.cvtColor(num_to_str(stage1_color_input), cv2.COLOR_BGR2HSV)

        hue_stage1_color = stage1_color_hsv[0][0][0]
        sat_stage1_color = stage1_color_hsv[0][0][1]
        val_stage1_color = stage1_color_hsv[0][0][2]


        lower_range = np.array([hue_stage1_color,100,100], dtype=np.uint8)
        upper_range = np.array([hue_stage1_color,255,255], dtype=np.uint8)


        # mask = cv2.inRange(hsv, lower_range, upper_range)
        mask = cv2.inRange(hsv, np.array([0,35,35]), np.array([20,255,255])) #  명도, 채도 확인 필요 
        result1 = cv2.bitwise_and(frame, frame, mask = mask)
        cv2.imshow("hsv", hsv)
        cv2.imshow("test", mask)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=10)

       

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # print(cnts)

        centre = None 

        for cnt in cnts:
            if len(cnt) > 0:
                c = max([cnt], key=cv2.contourArea)
                ((x,y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                centre = (int(M["m10"] / M["m00"]), int(M["m10"]/ M["m00"]))
                
                if radius > 10:
                    cv2.circle(frame, (int(x),int(y)), int(radius), (0,255,255),2)
                    cv2.circle(frame, centre, 5, (0,0,255),-1)
                    cv2.putText(frame, "{0}, {1}".format(x,y), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250,0,0), 2, cv2.LINE_AA)
                    print(f"[INFO] Object Center coordinates at X0 = {x} and Y0 = {y}")


        cv2.imshow('mask', mask)
        cv2.imshow('image', frame)
        k = cv2.waitKey(1)
        if(k == 27):
            break 
finally:
    connection.close()
    server_socket.close()
    cv2.destroyAllWindows()
