# Simulator ###########################################################################
#
# Copyright (c) 2021, Mohammad Rowshan
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, 
# are permitted provided that:
# the source code retains the above copyright notice, and te redistribtuion condition.
# 
# Freely distributed for educational and research purposes
#######################################################################################

from time import time
import numpy as np
import polar_coding_functions as pcf
from polar_code import PolarCode
from channel import channel
from rate_profile import rateprofile
from crclib import crc
import csv

N = 2**6
R = 0.5
crc_len = 0             # Use 8,12,16 along with the corresponding crc_poly below  # Use 0 for no CRC. # It does not support 6, 9, 11, 24, etc.
crc_poly = 0xA5         # 0x1021 for crc_len=16  # 0xC06  for crc_len=12 # 0xA6  for crc_len=8
list_size = 2**0        # Keep this as it is. Change list_size_max for your desired L.
list_size_max = 2**5    # For adaptive two-stage list decoding to accelerate the decoding process: first with L=list_size. If the deocing fails, then with L=list_size_max
designSNR = 2           # For instance for P(128,64):4, P(512,256):2
profile_name = "dega"   # Use "rm-polar" for Read-Muller polar code construction, #"pw" for polarization weight construction, "bh"  for Bhattachariya parameter/bound construction.

# For polar coding, set conv_gen = [1] which makes v=u meaninng no precoding.
conv_gen = [1,0,1,1,0,1,1]      # Use [1] for polar codes 

snrb_snr = 'SNRb'       # 'SNRb':Eb/N0 or 'SNR':Es/N0
modu = 'BPSK'           # It does not work for higher modulations

snr_range = np.arange(3,6,0.5) # in dB, (start,endpoint+step,step)
err_cnt = 50            # The number of error counts for each SNR point, for convergence, choose larger counts for low SNR regimes and for short codes.

systematic = False

# Error Coefficient-reduced: For code modifcation by X number of rows in G_N
# The maximum number of row modifications, you can choose 2 as well. If you increase it, the error coefficient might get betetr but the BLER may not. 
# See https://arxiv.org/abs/2111.08843
max_row_swaps  = 0      # 0 for no construction midifcation # 2 and 3 is recommended. 


#crc_len = len(bin(crc_poly)[2:].zfill(len(bin(crc_poly)[2:])//4*4+(len(bin(crc_poly)[2:])%4>0)*4))
K = int(N*R)
nonfrozen_bits = K + crc_len
mem = len(conv_gen)-1

rprofile = rateprofile(N,nonfrozen_bits,designSNR,max_row_swaps)

crc1 = crc(int(crc_len), crc_poly)
pcode = PolarCode(N, nonfrozen_bits, profile_name, L=list_size, rprofile=rprofile)
pcode.iterations = 10**7    # Maximum number of iterations if the errors found is less than err_cnt
pcode.list_size_max = list_size_max

print("PAC({0},{1}) constructed by {3}({4}dB)".format(N, nonfrozen_bits,crc_len,profile_name,designSNR))
print("L={} & c={}".format(list_size,conv_gen))
print("BER & FER evaluation is started")

st = time()
isCRCinc = True if crc_len>0 else False


class BERFER():
    """structure that keeps results of BER and FER tests"""
    def __init__(self):
        self.fname = str()
        self.label = str()
        self.snr_range = list()
        self.ber = list()
        self.fer = list()

result = BERFER()

pcode.m = mem
pcode.gen = conv_gen
pcode.cur_state = [0 for i in range(mem)]
log_M = 1   #M:modulation order



for snr in snr_range:
    print("\nSNR={0} dB".format(snr))
    ber = 0
    fer = 0
    ch = channel(modu, snr, snrb_snr, (K / N)) 
    for t in range(pcode.iterations):
        # Generating a K-bit binary pseudo-radom message
        #np.random.seed(t)
        message = np.random.randint(0, 2, size=K, dtype=int)
        if isCRCinc:
            message = np.append(message, crc1.crcCalc(message))

        x = pcode.pac_encode(message, conv_gen, mem, systematic)
        
        modulated_x = ch.modulate(x)
        y = ch.add_noise(modulated_x)

        llr_ch = ch.calc_llr3(y)

        decoded = pcode.pac_list_crc_decoder(llr_ch, 
                                            systematic,
                                            isCRCinc,
                                            crc1, 
                                            list_size)
        
        if pcf.fails(message, decoded)>0:
                pcode.edgeOrder = [0 for k in range(pcode.list_size_max)] #np.zeros(L, dtype=int)
                pcode.dLLRs = [0 for k in range(pcode.list_size_max)]
                pcode.PMs = [0 for k in range(pcode.list_size_max)]
                pcode.pathOrder = [0 for k in range(pcode.list_size_max)]
                decoded = pcode.pac_list_crc_decoder(llr_ch, systematic, isCRCinc, crc1, pcode.list_size_max)
        
        ber += pcf.fails(message, decoded)
        if not np.array_equal(message, decoded):
            fer += 1
            print("Error # {0} t={1}, FER={2:0.2e}".format(fer,t, fer/(t+1))) #"\nError #
        #fer += not np.array_equal(message, decoded)
        if fer > err_cnt:    #//:Floor Division
            print("@ {0} dB FER is {1:0.2e}".format(snr, fer/(t+1)))
            break
        if t%2000==0:
            print("t={0} FER={1} ".format(t, fer/(t+1)))
        if t==pcode.iterations:
            break
    #print("{0} ".format(ber))
    result.snr_range.append(snr)
    result.ber.append(ber / ((t + 1) * nonfrozen_bits))
    result.fer.append(fer / (t + 1))

    print("\n\n")
    print(result.label)
    print("SNR\t{0}".format(result.snr_range))
    print("FER\t{0}".format(result.fer))
    print("time on test = ", str(int((time() - st)/60)), ' min\n------------\n')


#Filename for saving the results
result.fname += "PAC({0},{1}),L{2},m{3}".format(N, pcode.nonfrozen_bits,list_size,mem)
if isCRCinc:
    result.fname += ",CRC{0}".format(crc_len)
    
#Writing the resuls in file
with open(result.fname + ".csv", 'w') as f:
    result.label = "PAC({0}, {1})\nL={2}\nRate-profile={3}\ndesign SNR={4}\n" \
                "Conv Poly={5}\nCRC={6} bits, Systematic={7}\n".format(N, pcode.nonfrozen_bits,
                pcode.list_size, profile_name, designSNR, conv_gen, crc_len, systematic)
    f.write(result.label)

    f.write("\nSNR: ")
    for snr in result.snr_range:
        f.write("{0}; ".format(snr))
    f.write("\nBER: ")
    for ber in result.ber:
        f.write("{0}; ".format(ber))
    f.write("\nFER: ")
    for fer in result.fer:
        f.write("{0}; ".format(fer))

print("\n\n")
print(result.label)
print("SNR\t{0}".format(result.snr_range))
print("BER\t{0}".format(result.ber))
#print("FER\t{0:1.2e}".format(result.fer))
print("FER\t{0}".format(result.fer))

print("time on test = ", str(time() - st), ' s\n------------\n')


"""
with open("bit_err_cnt.csv", 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter = ',', lineterminator = '\n') #Default:'\r\n' used in Unix # creating a csv writer object
        #csvwriter.writerow(row)
        csvwriter.writerows(map(lambda x: [x], pcode.bit_err_cnt[pcode.bitrev_indices]))"""
