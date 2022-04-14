Functions ##########################################################################
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

from operator import itemgetter
#itemgetter(item) return a callable object that fetches item from its operand using the operandâ€™s __getitem__() method. If multiple items are specified, returns a tuple of lookup values
import numpy as np
import math
from scipy.stats import norm



def fails(list1, list2):
    """returns number of bit errors"""
    return np.sum(np.absolute(list1 - list2))


def bitreversed(num: int, n) -> int:
    return int(''.join(reversed(bin(num)[2:].zfill(n))), 2)

    
# ------------ SC decoding functions -----------------
    
def lowerconv(upperdecision: int, upperllr: float, lowerllr: float) -> float:
    """PERFORMS IN LOG DOMAIN
    llr = lowerllr * upperllr - - if uppperdecision == 0
    llr = lowerllr / upperllr - - if uppperdecision == 1
    """
    if upperdecision == 0:
        return lowerllr + upperllr
    else:
        return lowerllr - upperllr


def logdomain_sum(x: float, y: float) -> float:
    if x < y:
        return y + np.log(1 + np.exp(x - y))
    else:
        return x + np.log(1 + np.exp(y - x))


def upperconv(llr1: float, llr2: float) -> float:
    """PERFORMS IN LOG DOMAIN
    llr = (llr1 * llr2 + 1) / (llr1 + llr2)"""
    #return logdomain_sum(llr1 + llr2, 0) - logdomain_sum(llr1, llr2)
    return np.sign(llr1)*np.sign(llr2)*min(abs(llr1),abs(llr2))


def logdomain_sum2(x, y):
    return np.array([x[i] + np.log(1 + np.exp(y[i] - x[i])) if x[i] >= y[i]
                     else y[i] + np.log(1 + np.exp(x[i] - y[i]))
                     for i in range(len(x))])

    
def upperconv2(llr1, llr2):
    """PERFORMS IN LOG DOMAIN
    llr = (llr1 * llr2 + 1) / (llr1 + llr2)"""
    return logdomain_sum2(llr1 + llr2, np.zeros(len(llr1))) - logdomain_sum2(llr1, llr2)



####Precoding for PAC COdes########################################

def conv_1bit(in_bit, cur_state, gen): 
    #This function calculates the 1 bit convolutional output during state transition
    g_len = len(gen)    #length of generator 
    g_bit = in_bit * gen[0]        

    for i in range(1,g_len):       
        if gen[i] == 1:
            #print(i-1,len(cur_state))
            #if i-1 > len(cur_state)-1 or i-1 < 0:
                #print("*****cur_state idex is {0} > {1}, g_len={2}".format(i-1,len(cur_state),g_len))
            g_bit = g_bit ^ cur_state[i-1]
    return g_bit



def getNextState(in_bit, cur_state, m):
#This function finds the next state during state transition
    #next_state = []
    if in_bit == 0:
        next_state = [0] + cur_state[0:m-1] # extend (the elements), not append
    else:
        next_state = [1] + cur_state[0:m-1]  #np.append([0], cur_state[0:m-1])     
    return next_state

def conv1bit_getNextStates(in_bit, cur_state1, cur_state2, gen1, gen2, bit_flag):
    m1 = len(gen1)-1
    m2 = len(gen2)-1

    g_bit = in_bit       

    if bit_flag == 1:
        for i in range(2,m1+1):       
            if gen1[i] == 1:
                g_bit = g_bit ^ cur_state1[i-1]
        for i in range(1,m2+1):       
            if gen2[i] == 1:
                g_bit = g_bit ^ cur_state2[i-1]
        if in_bit == 0:
            next_state2 = [0] + cur_state2[0:m2-1] # extend (the elements), not append
        else:
            next_state2 = [1] + cur_state2[0:m2-1]  #np.append([0], cur_state[0:m-1])
        if in_bit == 0:
            next_state1 = [0] + cur_state1[0:m1-1] # extend (the elements), not append
        else:
            next_state1 = [1] + cur_state1[0:m1-1]  #np.append([0], cur_state[0:m-1])
        #next_state1 = cur_state1
    else:
        for i in range(1,m1+1):       
            if gen1[i] == 1:
                g_bit = g_bit ^ cur_state1[i-1]
        for i in range(2,m2+1):       
            if gen2[i] == 1:
                g_bit = g_bit ^ cur_state2[i-1]
        if in_bit == 0:
            next_state1 = [0] + cur_state1[0:m1-1] # extend (the elements), not append
        else:
            next_state1 = [1] + cur_state1[0:m1-1]  #np.append([0], cur_state[0:m-1])     
        next_state2 = cur_state2
    
    return g_bit, next_state1, next_state2


def conv_encode(in_code, gen, m):
    # function to find the convolutional code for given input code (input code must be padded with zeros)
    #cur_state = np.zeros(m, dtype=np.int)         # intial state is [0 0 0 ...]
    cur_state = [0 for i in range(m)]#np.zeros(m, dtype=int)
    len_in_code = len(in_code)           # length of input code padded with zeros
    conv_code = np.zeros(len_in_code, dtype=int)     
    log_N = int(math.log2(len_in_code))
    for j in range(0,len_in_code):
        i = bitreversed(j, log_N)
        in_bit = in_code[i]              # 1 bit input 
        #if cur_state.size==0:
            #print("*****cur_state len is {0}, m={1}".format(cur_state.size,m))
        output = conv_1bit(in_bit, cur_state, gen);    # transition to next state and corresponding 2 bit convolution output
        cur_state = getNextState(in_bit, cur_state, m)    # transition to next state and corresponding 2 bit convolution output
        #conv_code = conv_code + [output]  #list   # append the 1 bit output to convolutional code
        conv_code[i] = output
    return conv_code


def bin2dec(binary): 
    decimal = 0
    for i in range(len(binary)): 
        decimal = decimal + binary[i] * pow(2, i) 
    return decimal







