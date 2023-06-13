import socket
import io 
import struct 
from PIL import Image 
import cv2
import numpy as np 
from glob import glob
import time


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
        im = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        cv2.imshow("test", im)
        cv2.waitKey(1)
       
        
        
finally:
    connection.close()
    server_socket.close()
