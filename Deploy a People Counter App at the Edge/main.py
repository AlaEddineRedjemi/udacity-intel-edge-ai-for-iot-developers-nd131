"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2 as cv # i'm too lazy to keep writing 'cv2' every time ^^

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.cpu_extension, args.device)
    ### TODO: Handle the input stream ###
    
    # check if the input is "camera" or "cam" strings
    if args.input.lower() == "cam" or args.input.lower() == "camera":
        args.input = 0
    elif os.path.splitext(args.input)[1] not in [".mp4", ".jpg", ".jpeg", ".png"]: # assume that we accept only mp4, jpg, jpeg, png
        print("Your input file is not supported! please provide a valid file.")
        return
    
    cap = cv.VideoCapture(args.input)
    cap.open(args.input)
    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    
    frame_counter = 0
    req_id = 0
    person_still_there = False
    total_count = 0
    duration = 0
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        ### TODO: Pre-process the image as needed ###
        p_frame = cv.resize(frame, (infer_network.get_input_shape()[3], infer_network.get_input_shape()[2]))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(p_frame, req_id)
        ### TODO: Wait for the result ###
        if infer_network.wait(req_id) == 0:
            #debugging ...
            #print("Latency :::: {:.2f}".format(infer_network.get_latency(frame_counter) / 1000))
            ### TODO: Get the results of the inference request ###
            rslt = infer_network.get_output(req_id)
            ### TODO: Extract any desired stats from the results ###
            frame_counter += 1
            current_count = 0

            for box in rslt[0][0]: # 1x1x100x7
                if box[2] >= prob_threshold:
                    xmin = int(box[3] * width)
                    ymin = int(box[4] * height)
                    xmax = int(box[5] * width)
                    ymax = int(box[6] * height)
                    cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                    current_count += 1
                    

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            
            if current_count != 0:
                client.subscribe("person")
                client.publish("person", json.dumps({"count" : current_count}))
                duration += 1 # accumulate all frames that contains persons to canculate seconds
                
                if frame_counter % fps == 0 and not person_still_there:
                    total_count += current_count
                    person_still_there = True
                    client.publish("person", json.dumps({"total" : total_count}))
                
                if duration % fps == 0:
                    client.subscribe("person/duration")
                    client.publish("person/duration", json.dumps({"duration" : (duration // fps) }))
            else:
                client.subscribe("person")
                client.publish("person", json.dumps({"count" : current_count}))
                person_still_there = False
                
            ### TODO: Send the frame to the FFMPEG server ###
            sys.stdout.buffer.write(frame)
            sys.stdout.flush()
            
            ### TODO: Write an output image if `single_image_mode` ###
            if os.path.splitext(args.input)[1] in [".jpg", ".jpeg", ".png"]:
                cv.imwrite("outputed"+os.path.splitext(args.input)[1], frame)
        
        # permutate req_id between 0 and 1 for every frame
        if req_id == 0:
            req_id = 1
        else:
            req_id = 0
            
    cap.release()
    cv.destroyAllWindows()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
