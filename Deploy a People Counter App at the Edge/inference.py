#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.network = None
        self.plugin = None
        self.exec_network = None
        self.cpu_ext = None
        self.in_blob = None
        self.out_blob = None
        self.infer_req = None
        
    def load_model(self, model, cpu_ext, device):
        ### TODO: Load the model ###
        if str(model).endswith(".xml"):
            m_xml = model
            m_bin = os.path.splitext(model)[0]+".bin"
        self.network = IENetwork(model=m_xml, weights=m_bin)
        ### TODO: Check for supported layers ###
        self.plugin = IECore()
        all_layers = self.network.layers.keys()
        supp_layers = self.plugin.query_network(self.network, device).keys()
        unsupp_layers = [l for l in all_layers if l not in supp_layers]
        
        ### TODO: Add any necessary extensions ###
        if len(unsupp_layers) != 0:
            if "CPU" in device:
                if cpu_ext:
                    self.plugin.add_extension(cpu_ext, device)
        ### TODO: Return the loaded inference plugin ###
        self.exec_network = self.plugin.load_network(self.network, device_name=device, num_requests=2)
        ### Note: You may need to update the function parameters. ###
        return self.exec_network

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        self.in_blob = next(iter(self.network.inputs))
        return self.network.inputs[self.in_blob].shape

    def exec_net(self, frame, req_id):
        ### TODO: Start an asynchronous request ###
        self.infer_req = self.exec_network.start_async(request_id=req_id, inputs={self.in_blob: frame})
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return self.infer_req

    def wait(self, req_id):
        ### TODO: Wait for the request to be complete. ###
        status = self.exec_network.requests[req_id].wait(-1)
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return status
    
    def get_latency(self, req_id):
        return self.exec_network.requests[req_id].latency

    def get_output(self, req_id):
        ### TODO: Extract and return the output results
        self.out_blob = next(iter(self.network.outputs))
        output = self.exec_network.requests[req_id].outputs[self.out_blob]
        ### Note: You may need to update the function parameters. ###
        return output
