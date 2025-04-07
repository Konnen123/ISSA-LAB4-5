import cv2
import numpy as np
import object_socket


s = object_socket.ObjectSenderSocket('127.0.0.1', 5000, print_when_awaiting_receiver=True, print_when_sending_object=True)

video = cv2.VideoCapture('.\\data\\Lane_Detection_Test_Video_01.mp4')

while True:
    ret, frame = video.read()
    s.send_object((ret, frame))

    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()

#
# for i in range(5):
#     a = np.random.randint(1, 9, (3, 3))
#     s.send(a)
#
#     print(f'\n--- {i} ---\n')
#     print(a)







