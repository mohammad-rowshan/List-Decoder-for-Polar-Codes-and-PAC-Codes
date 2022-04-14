# CRC8/12/16 Class ##############################################################
#
# Copyright (c) 2021, Mohammad Rowshan
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, 
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, 
#   this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# Freely distributed for educational and research purposes
###################################################################################

import numpy as np
#import math
"""
Created on Wed Nov  6 08:27:54 2019

@author: mrow0004
"""
class crc:
    #The __init__() function is called automatically 
    #every time the class is being used to create a new object.
    #he self parameter is a reference to the current instance of the class, 
    #and is used to access variables that belong to the class.
    #It does not have to be named self, 
    #but it has to be the first parameter of any function in the class.
    def __init__(self, crc_len, crc_poly):
        self.len = crc_len
        self.gen = crc_poly
        self.crc_table = self.build_crc_table()

    def int_to_binlist(self, num: int, size: int):
        return [int(bit) for bit in bin(num)[2:].zfill(size)]
    
    def build_crc8_table(self):
        generator = self.gen
        crc8_table = list()
        for div in range(256):
            cur_byte = div #np.uint8(div << 8)
            for bit in range(8):
                #temp1 = np.bitwise_and(cur_byte, np.uint16(0x8000))
                if np.bitwise_and(cur_byte, np.uint8(0x80)) != np.uint8(0x00):
                    # cur_byte = np.left_shift(cur_byte, 1)  #
                    cur_byte <<= 1
                    # cur_byte = np.bitwise_xor(cur_byte, generator)  #
                    cur_byte ^= generator
                else:
                    # cur_byte = np.left_shift(cur_byte, 1)  #
                    cur_byte <<= 1
            crc8_table.append(np.uint8(cur_byte))
        return crc8_table

    def build_crc12_table(self):
        """"""
        generator = self.gen
        crc12_table = []
        for div in range(256):
            cur_byte = div << 4
            for bit in range(8):
                cur_byte <<= 1
                if cur_byte & 0x1000:
                #if np.bitwise_and(cur_byte, 0x800) != 0x000:
                    cur_byte ^= generator
                    pass
                continue
            crc12_table.append(cur_byte & 0xfff)
        return crc12_table
    
    def build_crc16_table(self):
        """"""
        generator = self.gen
        crc16_table = list()
        for div in range(256):
            cur_byte = np.uint16(div << 8)
            for bit in range(8):
                #temp1 = np.bitwise_and(cur_byte, np.uint16(0x8000))
                if np.bitwise_and(cur_byte, np.uint16(0x8000)) != np.uint16(0x0000):
                    # cur_byte = np.left_shift(cur_byte, 1)  #
                    cur_byte <<= 1
                    # cur_byte = np.bitwise_xor(cur_byte, generator)  #
                    cur_byte ^= generator
                else:
                    # cur_byte = np.left_shift(cur_byte, 1)  #
                    cur_byte <<= 1
            crc16_table.append(np.uint16(cur_byte))
        return crc16_table

    def crc8_table_method(self, info):
        """"""
        crc = 0
        if info.size%8 != 0:
            pad0 = np.zeros((info.size//8*8+8)-info.size, dtype=np.int8)
            info = np.append(pad0, info)
        # Byte-oriented: Used for packing every 8 bits (one byte), because data are stored in bytes
        coef = np.array([128, 64, 32, 16, 8, 4, 2, 1])  # for easy left shift by 8
        for b in range(0, len(info), 8):
            pos = np.uint8((crc) ^ np.sum(info[b:b+8] * coef))
            crc = self.crc_table[pos]
        return self.int_to_binlist(crc, 8)
    
   
    def crc12_table_method(self, info):
        """"""
        crc = 0
        # Byte-oriented: Used for packing every 8 bits (one byte), because data are stored in bytes
        if info.size%8 != 0:
            pad0 = np.zeros((info.size//8*8+8)-info.size, dtype=np.int8)
            info = np.append(pad0, info)
        coef = np.array([128, 64, 32, 16, 8, 4, 2, 1])  # for easy left shift by 8
        for b in range(0, len(info), 8):
            #pos = np.uint8((crc >> 4) ^ np.sum(info[b:b+8] * coef))
            #crc = np.uint16((crc << 8) ^ crc_table[pos])
            pos = ((crc >> 4) & 0xff) ^ np.sum(info[b:b+8] * coef)
            crc = ((crc << 8) & 0xfff) ^ self.crc_table[pos]
        #print(crc)
        return self.int_to_binlist(crc, 12)
        
    
    def crc16_table_method(self, info):
        """"""
        crc = 0
        if info.size%8 != 0:
            pad0 = np.zeros((info.size//8*8+8)-info.size, dtype=np.int8)
            info = np.append(pad0, info)
        coef = np.array([128, 64, 32, 16, 8, 4, 2, 1])  # for easy left shift by 8
        for b in range(0, len(info), 8):
            pos = np.uint16((crc >> 8) ^ np.sum(info[b:b+8] * coef))
            crc = np.uint16((crc << 8) ^ self.crc_table[pos])
        return self.int_to_binlist(crc, 16)

    def build_crc_table(self):
        if self.len == 8:
            return self.build_crc8_table()
        elif self.len == 12:
            return self.build_crc12_table()
        elif self.len == 16:
            return self.build_crc16_table()
            
    def crcCalc(self, info):
        if self.len == 8:
            return self.crc8_table_method(info)
        elif self.len == 12:
            return self.crc12_table_method(info)
        elif self.len == 16:
            return self.crc16_table_method(info)
            
            
