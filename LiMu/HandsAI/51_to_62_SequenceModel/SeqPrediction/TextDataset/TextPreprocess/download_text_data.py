# -*- coding: utf-8 -*-
# @Time    : 2024/3/6 15:16
# @Author  : Karry Ren

""" Download the time-machine from web for using. """

import d2l.torch as d2l

d2l.DATA_HUB["time_machine"] = (d2l.DATA_URL + "timemachine.txt", "090b5e7e70c295757f55df93cb0a180b9691891a")
d2l.download("time_machine")
