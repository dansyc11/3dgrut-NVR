#!/usr/bin/env python3

import os
import sys
import time

import argparse
import numpy as np

#sys.path.insert(0, '/usr/lib/python3/dist-packages')
import ecal.core.core as ecal_core

sys.path.append('/opt/vilota/bin')
sys.path.append('/opt/vilota/python')
import os
import sys
import time

import capnp
import numpy as np
import cv2


sys.path.append('/opt/vilota/bin')
sys.path.append('/opt/vilota/python')



import ecal.core.core as ecal_core
from capnp_subscriber import CapnpSubscriber

# pycapnp version >= 2.0
sys.path.append(os.path.join(os.path.dirname(__file__), '../vk_sdk/capnp'))
sys.path.append('/opt/vilota/messages')
capnp.add_import_hook()

import image_capnp as eCALImage

imshow_map = {}


def addStats(image, imageMsg):
    brightness = imageMsg.mipMapBrightness
    brightnessChange = imageMsg.mipMapBrightnessChange

    # Define the text and font properties
    text = f"b: {brightness} c:{brightnessChange:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    font_color = (255, 255, 255)  # White color

    text_position = (10, 30)


#    cv2.putText(image, text, text_position, font, font_scale, font_color, font_thickness)


def callback(type, topic_name, msg, ts):
    # need to remove the .decode() function within the Python API of ecal.core.subscriber StringSubscriber
    with eCALImage.Image.from_bytes(msg) as imageMsg:
        print(f"SN = {imageMsg.header.frameId}")
        print(
            f"seq = {imageMsg.header.seq}, stamp = {imageMsg.header.stampMonotonic}, with {len(msg)} bytes, encoding = {imageMsg.encoding}")
        # print(f"latency device = {imageMsg.header.latencyDevice / 1e6} ms")
        # print(f"latency host = {imageMsg.header.latencyHost / 1e6} ms")
        print(f"width = {imageMsg.width}, height = {imageMsg.height}, mipmap = {imageMsg.mipMapLevels}")
        print(f"exposure = {imageMsg.exposureUSec}, gain = {imageMsg.gain}")
        print(f"intrinsic = {imageMsg.intrinsic}")
        # print(f"extrinsic = {imageMsg.extrinsic}")
        print(f"instant w = {imageMsg.motionMeta.instantaneousAngularVelocity}")
        print(f"average w = {imageMsg.motionMeta.averageAngularVelocity}")

        if (imageMsg.encoding == "mono8"):

            mat = np.frombuffer(imageMsg.data, dtype=np.uint8)
            mat = mat.reshape((imageMsg.height, imageMsg.width, 1))

            addStats(mat, imageMsg)

            imshow_map[topic_name + " mono8"] = mat

            # cv2.imshow("mono8", mat)
            # cv2.waitKey(3)
        elif (imageMsg.encoding == "yuv420"):
            mat = np.frombuffer(imageMsg.data, dtype=np.uint8)
            mat = mat.reshape((imageMsg.height * 3 // 2, imageMsg.width, 1))

            mat = cv2.cvtColor(mat, cv2.COLOR_YUV2BGR_IYUV)

            imshow_map[topic_name + " yuv420"] = mat
            # cv2.imshow("yuv420", mat)
            # cv2.waitKey(3)
        elif (imageMsg.encoding == "bgr8"):
            mat = np.frombuffer(imageMsg.data, dtype=np.uint8)
            mat = mat.reshape((imageMsg.height, imageMsg.width, 3))
            imshow_map[topic_name + " bgr8"] = mat
        elif (imageMsg.encoding == "jpeg"):
            mat_jpeg = np.frombuffer(imageMsg.data, dtype=np.uint8)
            mat = cv2.imdecode(mat_jpeg, cv2.IMREAD_GRAYSCALE)
            imshow_map[topic_name + " jpeg"] = mat
        else:
            raise RuntimeError("unknown encoding: " + imageMsg.encoding)


def main():
    # mat = np.ones((800,1280,1), dtype=np.uint8) * 125
    # cv2.imshow("mono8", mat)
    # cv2.imwrite("test0.jpg", mat)

    # print eCAL version and date
    print("eCAL {} ({})\n".format(ecal_core.getversion(), ecal_core.getdate()))

    # initialize eCAL API
    ecal_core.initialize(sys.argv, "test_image_sub")

    # set process state
    ecal_core.set_process_state(1, 1, "I feel good")

    # create subscriber and connect callback

    n = len(sys.argv)
    topic_string=""
    if n == 1:
        topics = ["S0/cama"]
    elif n >= 2:
        topics = []
        for i in range(1, n):
            topics.append(sys.argv[i])
            print(f"topic {i} = {topics[-1]}")
    else:
        raise RuntimeError("Need to pass in exactly one parameter for topic")
    
    for topic in topics:
        topic_string += topic + ","
    
    subs = []
    for topic in topics:
        print(f"Streaming topic {topic}")
        sub = CapnpSubscriber("Image", topic)
        sub.set_callback(callback)
        subs.append(sub)

    time.sleep(2)
    # idle main thread
    cv2.namedWindow(topic_string, flags=cv2.WINDOW_GUI_NORMAL)
    while ecal_core.ok():
        im_maps = []

        scale_percent = 100 #int(sys.argv[1])  # percent of original size
        width = int(1280 * scale_percent / 100)  # im.shape[1]
        height = int(800 * scale_percent / 100)  # im.shape[0]
        dim = (width, height) #(1280, 800)

        for im in imshow_map:
            # resize image to standardise
            imshow_map[im] = cv2.resize(imshow_map[im], dim, interpolation=cv2.INTER_AREA)
            im_maps.append(imshow_map[im])

        if len(imshow_map) == 1:
            cv2.imshow(topic_string, im_maps[0])
        elif len(imshow_map) == 2: # stack vertically
            image_stack_tot = np.vstack((im_maps[0], im_maps[1]))
            cv2.imshow(topic_string, image_stack_tot)
        elif len(imshow_map) <= 4:
            while len(im_maps) <= 3:
                im_maps.append(np.zeros((height, width), dtype=np.uint8))

            image_stack_1 = np.hstack((im_maps[0], im_maps[1]))
            image_stack_2 = np.hstack((im_maps[2], im_maps[3]))
            image_stack_tot = np.vstack((image_stack_1, image_stack_2))
            cv2.imshow(topic_string, image_stack_tot)
        else:
            pass # not showing any thing if more than 4 images

        cv2.waitKey(10)


    # finalize eCAL API
    ecal_core.finalize()



if __name__ == "__main__":
    main()

