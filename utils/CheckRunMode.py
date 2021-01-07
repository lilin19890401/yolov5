#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2021/1/4 16:15
# @Author       : LiLin
# @File         : CheckRunMode.py
# @Software     : PyCharm
# @Description  :
import sys

def IS_Debug():
    gettrace = getattr(sys, 'gettrace',None)
    if gettrace is None:
        return False
    elif gettrace():
        return True
    else:
        return False


if __name__ == '__main__':
    print(IS_Debug())
