# AWGN Channel Class ##############################################################
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

class channel:
    def __init__(self, modulation, snrdB, snrb_snr, Rc):
        self.modulation = modulation
        self.M = 4 if modulation.upper() == 'QPSK' else 2
        self.noise_power = self.calc_N0(snrdB, snrb_snr, Rc)
        self.code_word_length = 0
        #self.sigma = np.sqrt(N0/2)
        #self.snrb_snr = snrb_snr
        self.constell = self.construct_mpsk(self.M, rotate=False)
        self.subconstells = self.get_subconstells(self.constell)
        
    def calc_N0(self, snrdB, snrb_snr, Rc):
        #Rc = (K / N)
        if snrb_snr.upper() == 'SNR':
            return 1 / pow(10, snrdB / 10)  #Noise power
        else:
            return 1 / (np.log2(self.M)*Rc*pow(10, snrdB / 10))  #Noise power, Linear SNR: pow(10, snrdB / 10)

    def modulate(self, m):
        modulated = []
        self.code_word_length = len(m)
        if self.modulation.upper() == 'BPSK':
            modulated = [(1 - 2 * x) for x in m]
        elif self.modulation.upper() == 'QPSK':
            if (np.mod(len(m),2)): # Zero Padding
                m = [0] + list(m)
            for msb, lsb in zip(m[0::2], m[1::2]): # Traversing Lists in Parallel : The zip() function takes iterables, aggregates them in a tuple, and return it.
                modulated.append( 1/np.sqrt(2) * (1 * (1 + 1j) - 2 * (msb + lsb*1j) ))
        return modulated


    def add_noise(self, signal):
        if self.modulation.upper() == 'BPSK':
            #return signal + noise_power*np.random.randn(len(signal))
            return signal + np.sqrt(self.noise_power/2) * np.random.standard_normal(len(signal))
        elif self.modulation.upper() == 'QPSK':
            return signal + np.sqrt(self.noise_power/2) * np.random.randn(len(signal)) * (1 + 1j)
            #return signal + self.noise_power/np.sqrt(2) * np.random.randn(len(signal)) + self.noise_power/np.sqrt(2) * np.random.randn(len(signal)) * 1j

    def calc_llr(self,c):
        llr = []
        if self.modulation.upper() == 'BPSK':
            llr = [(4/self.noise_power*y) for y in c]
        elif self.modulation.upper() == 'QPSK':
            for y in c:
                llr.extend([(4*y.real/self.noise_power),(4*y.imag/self.noise_power)])
                #llr.extend([(4*np.sqrt(2)**y.real/self.noise_power),(4*np.sqrt(2)*y.imag/self.noise_power)])
                #llr += [(-4/np.sqrt(2)*y.real/self.noise_power),(-4/np.sqrt(2)*y.imag/self.noise_power)]
            """llr = np.reshape(np.transpose([4 * np.imag(c) / self.noise_power,
                                            4 * np.real(c) / self.noise_power]), 1*len(c))"""

        return np.array(llr)




    def calc_llr2(self, c):
        """       """
        llr0 = []
        llr = []
        if self.modulation.upper() == 'BPSK':
            llr = [(4/self.noise_power*y) for y in c]
        elif self.modulation.upper() == 'QPSK':
            a = 0.70710678
            MSB_set = [
                    [-a-a*1j, a-a*1j], [-a+a*1j, a+a*1j]
                    ]
            LSB_set = [
                    [-a+a*1j, -a-a*1j], [a+a*1j, a-a*1j]
                    ]
            for y in c:
                L_MSB = 1 / self.noise_power * ( min( (y.real - MSB_set[0][0].real)**2 + (y.imag - MSB_set[0][0].imag)**2, (y.real - MSB_set[0][1].real)**2 + (y.imag - MSB_set[0][1].imag)**2 ) - min( (y.real - MSB_set[1][0].real)**2 + (y.imag - MSB_set[1][0].imag)**2, (y.real - MSB_set[1][1].real)**2 + (y.imag - MSB_set[1][1].imag)**2 ) )
                L_LSB = 1 / self.noise_power * ( min( (y.real - LSB_set[0][0].real)**2 + (y.imag - LSB_set[0][0].imag)**2, (y.real - LSB_set[0][1].real)**2 + (y.imag - LSB_set[0][1].imag)**2 ) - min( (y.real - LSB_set[1][0].real)**2 + (y.imag - LSB_set[1][0].imag)**2, (y.real - LSB_set[1][1].real)**2 + (y.imag - LSB_set[1][1].imag)**2 ) )
                llr.extend([L_LSB, L_MSB])
                llr0.extend([(4*y.real/self.noise_power),(4*y.imag/self.noise_power)])
        return np.array(llr)




    def calc_llr3(self, c):
        """       """
        llr0 = []
        llr = []
        if self.modulation.upper() == 'BPSK':
            llr = [(4/self.noise_power*y) for y in c]
        elif self.modulation.upper() == 'QPSK':
            a = 0.70710678
            MSB_set = [
                    [-a-a*1j, a-a*1j], [-a+a*1j, a+a*1j]
                    ]
            LSB_set = [
                    [-a+a*1j, -a-a*1j], [a+a*1j, a-a*1j]
                    ]
            for y in c:
                L_MSB =  -np.log(( np.exp( -1 / self.noise_power *( (y.real - MSB_set[0][0].real)**2 + (y.imag - MSB_set[0][0].imag)**2 )) +  np.exp( -1 / self.noise_power *( (y.real - MSB_set[0][1].real)**2 + (y.imag - MSB_set[0][1].imag)**2 )) ) /  ( np.exp( -1 / self.noise_power *( (y.real - MSB_set[1][0].real)**2 + (y.imag - MSB_set[1][0].imag)**2 )) +  np.exp( -1 / self.noise_power *((y.real - MSB_set[1][1].real)**2 + (y.imag - MSB_set[1][1].imag)**2 )) ))
                L_LSB =  -np.log(( np.exp( -1 / self.noise_power *( (y.real - LSB_set[0][0].real)**2 + (y.imag - LSB_set[0][0].imag)**2 )) +  np.exp( -1 / self.noise_power *( (y.real - LSB_set[0][1].real)**2 + (y.imag - LSB_set[0][1].imag)**2 )) ) /  ( np.exp( -1 / self.noise_power *( (y.real - LSB_set[1][0].real)**2 + (y.imag - LSB_set[1][0].imag)**2 )) +  np.exp( -1 / self.noise_power *((y.real - LSB_set[1][1].real)**2 + (y.imag - LSB_set[1][1].imag)**2 )) ))
                llr.extend([L_LSB, L_MSB])
                llr0.extend([(4*y.real/self.noise_power),(4*y.imag/self.noise_power)])
        return np.array(llr)




    def construct_mpsk(self, m, rotate=True):
        """Function to build M-PSK constellation values"""
        if m == 2:
            return np.array([1, -1])
        ref_i = np.cos(np.array([i for i in range(m)]) / m * 2 * np.pi + rotate * np.pi / m) # Eqivalent to 1/sqrt(2)
        ref_q = np.sin(np.array([i for i in range(m)]) / m * 2 * np.pi + rotate * np.pi / m)
        return ref_i + 1j * ref_q


    def get_subconstells(self, constell: np.ndarray):
        """Function to build sub constellations for further llr detection"""
        order = np.log2(len(constell))
        positions = np.arange(len(constell))
        return np.array([[[constell[(positions >> i) % 2 == j]] for j in range(2)] for i in range(int(order))])
        # positions >> 1 : array([0, 0, 1, 1], dtype=int32) #(positions >> 1) % 2 == 1: array([False, False,  True,  True]) # (positions >> 1) % 2 == 0: array([ True,  True, False, False])

    def sum_num_denum(self, rx: complex):
        """       """
        zer = [np.exp((np.real(rx) * np.transpose(np.real(self.subconstells[i][0])) + #each element has two sub elements
                       np.imag(rx) * np.transpose(np.imag(self.subconstells[i][0])) / self.noise_power)).sum(axis=0)
               for i in range(int(np.log2(self.M)))]
        one = [np.exp((np.real(rx) * np.transpose(np.real(self.subconstells[i][1])) +
                       np.imag(rx) * np.transpose(np.imag(self.subconstells[i][1])) / self.noise_power)).sum(axis=0)
               for i in range(int(np.log2(self.M)))]
        return np.array([zer, one])        

    def calc_llr2_(self, c):
        """ """
        precounted = self.sum_num_denum(c)
        llrs = np.log(precounted[0] / precounted[1])
        llrs = np.reshape(np.transpose(llrs), llrs.size)
        return llrs


