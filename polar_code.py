# Polar Code Class incldung Decoding Method ###########################################
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

import polar_coding_exceptions as pcexc
import polar_coding_functions as pcfun
import copy
import numpy as np
import csv
import math
#from rate_profile import rateprofile


class path:
    """A single branch entailed to a path for list decoder"""
    #These branches represent the paths as well
    def __init__(self, N=128, m=6):
        self.N = N  # codeword length
        self.n = int(pcfun.np.log2(N))  # number of levels
        self.llrs = pcfun.np.zeros(2 * self.N - 1)
        self.bits = pcfun.np.zeros((2, self.N-1), dtype=int) # The partial sums 
        self.decoded = pcfun.np.zeros(self.N, dtype=int) # The results of PAC decoding
        self.polar_decoded = pcfun.np.zeros(self.N, dtype=int)  # The intermediate results of Polar decoding
        self.corrprob = 0  # Path Metric
        self.forkprob = 0  # probability for forking
        self.forkval = 0  # value for forking

        self.isCorrPath = 1  # Is this the correct path?
        self.pathOrder = 0  # Path Order

        self.cur_state = [0 for i in range(m)]  # Current state

    def __repr__(self):
        return repr((self.llrs, self.bits, self.decoded, self.corrprob, self.forkval, self.forkprob))



    def update_llrs(self, position: int):
        if position == 0:
            nextlevel = self.n
        else:
            lastlevel = (bin(position)[2:].zfill(self.n)).find('1') + 1
            start = int(pcfun.np.power(2, lastlevel - 1)) - 1
            end = int(pcfun.np.power(2, lastlevel) - 1) - 1
            for i in range(start, end + 1):
                self.llrs[i] = pcfun.lowerconv(self.bits[0][i],
                                               self.llrs[end + 2 * (i - start) + 1],
                                               self.llrs[end + 2 * (i - start) + 2])
            nextlevel = lastlevel - 1
        for lev in range(nextlevel, 0, -1):
            start = int(pcfun.np.power(2, lev - 1)) - 1
            end = int(pcfun.np.power(2, lev) - 1) - 1
            for indx in range(start, end + 1):
                exp1 = end + 2 * (indx - start)
                llr1 = self.llrs[exp1 + 1]
                llr2 = self.llrs[exp1 + 2]
                #self.llrs[indx] = pcfun.upperconv(self.llrs[exp1 + 1], self.llrs[exp1 + 2])
                #SPCparams[irs].LLR[indx] = SIGN(llr1)*SIGN(llr2)*(float)min(fabs(llr1), fabs(llr2));
                self.llrs[indx] = np.sign(llr1)*np.sign(llr2)*min(abs(llr1),abs(llr2))
                #intLLR = self.llrs[indx]
    def update_bits(self, position: int):
        N = self.N
        latestbit = self.polar_decoded[position]
        #print("d{0}".format(self.decoded[position]))
        n = self.n
        if position == N - 1:
            return
        elif position < N // 2:
            self.bits[0][0] = latestbit
        else:
            lastlevel = (bin(position)[2:].zfill(n)).find('0') + 1
            self.bits[1][0] = latestbit
            for lev in range(1, lastlevel - 1):
                st = int(pcfun.np.power(2, lev - 1)) - 1
                ed = int(pcfun.np.power(2, lev) - 1) - 1
                for i in range(st, ed + 1):
                    self.bits[1][ed + 2 * (i - st) + 1] = (self.bits[0][i] + self.bits[1][i]) % 2
                    self.bits[1][ed + 2 * (i - st) + 2] = self.bits[1][i]

            lev = lastlevel - 1
            st = int(pcfun.np.power(2, lev - 1)) - 1
            ed = int(pcfun.np.power(2, lev) - 1) - 1
            for i in range(st, ed + 1):
                self.bits[0][ed + 2 * (i - st) + 1] = (self.bits[0][i] + self.bits[1][i]) % 2
                self.bits[0][ed + 2 * (i - st) + 2] = self.bits[1][i]
        #print("s{0}".format(self.bits[0][0]))
            
    def update_corrprob(self):
        self.corrprob += self.forkprob
        #self.corrprob *= self.forkprob


class PolarCode:
    """Represent constructing polar codes,
    encoding and decoding messages with polar codes"""

    def __init__(self, N, K, construct, L, rprofile):
        if K > N: #K >= N:
            raise pcexc.PCLengthError
        elif pcfun.np.log2(N) != int(pcfun.np.log2(N)):
            raise pcexc.PCLengthDivTwoError
        else:
            self.codeword_length = N
            self.log2_N = int(math.log2(N))
            self.nonfrozen_bits = K
            #self.designSNR = dSNR
            self.n = int(pcfun.np.log2(self.codeword_length))
            #self.bitrev_indices = np.array([pcfun.bitreversed(j, self.n) for j in range(N)])
            self.bitrev_indices = [pcfun.bitreversed(j, self.n) for j in range(N)]
            #self.polarcode_mask = pcfun.rm_build_mask(N, K, dSNR) if construct=="rm" else pcfun.RAN87_build_mask(N, K, dSNR) if  construct=="ran87" else pcfun.build_mask(N, K, dSNR)
            self.rprofile = rprofile
            self.polarcode_mask = self.rprofile.build_mask(construct) #in bit-reversal order
            self.polarcode_mask = self.rprofile.modify_profile() 
            self.rate_profile = self.polarcode_mask[self.bitrev_indices] #in decoding order
            self.frozen_bits = (self.polarcode_mask + 1) % 2  #in bitrevesal order
            self.LLRs = np.zeros(2 * self.codeword_length - 1, dtype=float)
            self.BITS = np.zeros((2, self.codeword_length - 1), dtype=int)
            self.stem_LLRs = np.zeros(2 * self.codeword_length - 1, dtype=float)
            self.stem_BITS = np.zeros((2, self.codeword_length - 1), dtype=int)
            self.list_size = L
            self.list_size_max = L
            self.curr_list_size = 1
            self.sc_list = list()
            self.edgeOrder = [0 for k in range(L)] #np.zeros(L, dtype=int)
            self.PMs = [0 for k in range(L)]
            self.pathOrder = [0 for k in range(L)]
            self.iterations = 10**6

            
            self.m = 0
            self.gen = []
            self.cur_state = [] #np.zeros(self.m, dtype=int)#
            #list([iterbale]) is the list constructor
            self.modu = 'BPSK'
            
            
            self.sigma = 0
            self.snrb_snr = 'SNRb'
            
            self.bits_contrib_in_mhd = np.zeros(len(self.rprofile.rows_wt(self.rprofile.min_row_wt())), dtype=int)
            self.wt_distrib = np.zeros(self.codeword_length, dtype=int)


    def __repr__(self):
        return repr((self.codeword_length, self.nonfrozen_bits, self.designSNR))
#__str__ (read as "dunder (double-underscore) string") and __repr__ (read as "dunder-repper" (for "representation")) are both special methods that return strings based on the state of the object.
    def mul_matrix(self, profiled):
        """multiplies message of length N with generator matrix G"""
        """Multiplication is based on factor graph"""
        N = self.codeword_length
        polarcoded = profiled
        for i in range(self.n):
            if i == 0:
                polarcoded[0:N:2] = (polarcoded[0:N:2] + polarcoded[1:N:2]) % 2
            elif i == (self.n - 1):
                polarcoded[0:int(N/2)] = (polarcoded[0:int(N/2)] + polarcoded[int(N/2):N]) % 2
            else:
                enc_step = int(pcfun.np.power(2, i))
                for j in range(enc_step):
                    polarcoded[j:N:(2 * enc_step)] = (polarcoded[j:N:(2 * enc_step)]
                                                    + polarcoded[j + pcfun.np.power(2, i):N:(2 * enc_step)]) % 2
        return polarcoded
    # --------------- ENCODING -----------------------

    def profiling(self, info):
        """Apply polar code mask to information message and return profiled message"""
        profiled = pcfun.np.zeros(self.codeword_length, dtype=int) #array
        profiled[self.polarcode_mask == 1] = info
        self.trdata = copy.deepcopy(profiled)
        return profiled

   

    def encode(self, info, issystematic: bool):
        """Encoding function"""
        # Non-systematic encoding
        encoded = self.profiling(info)
        if not issystematic:
            polarcoded = self.mul_matrix(encoded)
        # Systematic encoding based on non-systematic encoding
        else:
            polarcoded = self.mul_matrix(encoded)
            polarcoded *= self.polarcode_mask
            polarcoded = self.mul_matrix(polarcoded)
            # ns_encoded = self.mul_matrix(self.profiling(info))
            # s_encoded = [self.polarcode_mask[i] * ns_encoded[i] for i in range(self.codeword_length)]
            # return self.mul_matrix(s_encoded)
        return polarcoded



    def pac_encode(self, info, conv_gen, mem):
        """Encoding function"""
        # Non-systematic encoding
        V = self.profiling(info)
        U = pcfun.conv_encode(V, conv_gen, mem)
        X = self.mul_matrix(U)
        return X


    # -------------------------- DECODING -----------------------------------
    def extract(self, decoded_message):
        """Extracts bits from information positions due to polar code mask"""
        decoded_info = pcfun.np.array(list(), dtype=int)
        mask = self.polarcode_mask
        for i in range(len(self.polarcode_mask)):
            if mask[i] == 1:
                decoded_info = pcfun.np.append(decoded_info, decoded_message[i])
        return decoded_info


    # --- LIST Decoding ---------------------------------------------------------
    def pac_fork(self, sc_list, pos):
        """forks current stage of SCList decoding
        and makes decisions on decoded values due to llr values"""
        pos_rev = pcfun.bitreversed(pos,self.n)
        TxMsg = self.trdata[pos]
        edgeValue = [0 for i in range(2*self.curr_list_size)]   #encoded by CE
        msgValue = [0 for i in range(2*self.curr_list_size)]    #Msg bit
        pathMetric = [0.0 for i in range(2*self.curr_list_size)]
        pathState = [[] for i in range(2*self.curr_list_size)]

        for i in range(self.curr_list_size):
            i2 = i+self.curr_list_size
            if sc_list[i].llrs[0] > 0:
                edgeValue[i] = pcfun.conv_1bit(0, sc_list[i].cur_state, self.gen)
                edgeValue[i2] = 1 - edgeValue[i]
                pathMetric[i] = sc_list[i].corrprob + (0 if edgeValue[i]==0 else 1) * np.abs(sc_list[i].llrs[0])
                pathMetric[i2] = sc_list[i].corrprob + (0 if edgeValue[i2]==0 else 1) * np.abs(sc_list[i].llrs[0])
                if pathMetric[i2] > pathMetric[i]:
                    msgValue[i] = 0
                    msgValue[i2] = 1
                    pathState[i] = pcfun.getNextState(0, sc_list[i].cur_state, self.m)
                    pathState[i2] = pcfun.getNextState(1, sc_list[i].cur_state, self.m)
                else:
                    edgeValue[i] = 1 - edgeValue[i]
                    edgeValue[i2] = 1 - edgeValue[i2]
                    tempPM = pathMetric[i]
                    pathMetric[i] = pathMetric[i2]
                    pathMetric[i2] = tempPM
                    msgValue[i] = 1
                    msgValue[i2] = 0
                    pathState[i] = pcfun.getNextState(1, sc_list[i].cur_state, self.m)
                    pathState[i2] = pcfun.getNextState(0, sc_list[i].cur_state, self.m)
                    
            else:

                edgeValue[i] = pcfun.conv_1bit(1, sc_list[i].cur_state, self.gen)
                edgeValue[i2] = 1 - edgeValue[i]
                pathMetric[i] = sc_list[i].corrprob + (0 if edgeValue[i]==1 else 1) * np.abs(sc_list[i].llrs[0])
                pathMetric[i2] = sc_list[i].corrprob + (0 if edgeValue[i2]==1 else 1) * np.abs(sc_list[i].llrs[0])

                if pathMetric[i2] > pathMetric[i]:  #to avoid deepcopy in SC state (not helpful in deletion or duplicate states)
                    msgValue[i] = 1
                    msgValue[i2] = 0
                    pathState[i] = pcfun.getNextState(1, sc_list[i].cur_state, self.m)
                    pathState[i2] = pcfun.getNextState(0, sc_list[i].cur_state, self.m)
                else:
                    edgeValue[i] = 1 - edgeValue[i]
                    edgeValue[i2] = 1 - edgeValue[i2]
                    tempPM = pathMetric[i]
                    pathMetric[i] = pathMetric[i2]
                    pathMetric[i2] = tempPM
                    msgValue[i] = 0
                    msgValue[i2] = 1
                    pathState[i] = pcfun.getNextState(0, sc_list[i].cur_state, self.m)
                    pathState[i2] = pcfun.getNextState(1, sc_list[i].cur_state, self.m)
        
        PM_sorted_idx = np.argsort(pathMetric, kind='mergesort')
        sortedPM_idx = np.argsort(pathMetric)
        if self.curr_list_size < self.list_size:
            self.edgeOrder = copy.deepcopy(PM_sorted_idx[:2*self.curr_list_size])

            for i in range(self.curr_list_size):
                i2 = i+self.curr_list_size
                copy_branch = path(self.codeword_length, self.m)
                #If we don't use deepcopy, the new object will refer to the original one and works as a pointer
                copy_branch = copy.deepcopy(sc_list[i])
                sc_list[i].corrprob = pathMetric[i]
                sc_list[i].decoded[pos] = msgValue[i]
                sc_list[i].polar_decoded[pos] = edgeValue[i]
                sc_list[i].cur_state = pathState[i]
                copy_branch.corrprob = pathMetric[i2]
                copy_branch.decoded[pos] = msgValue[i2]
                copy_branch.polar_decoded[pos] = edgeValue[i2]
                copy_branch.cur_state = pathState[i2]
                if sc_list[i].isCorrPath == 1:
                    sc_list[i].isCorrPath = 1 if self.trdata[pos] == msgValue[i] else 0
                    copy_branch.isCorrPath = 1 if self.trdata[pos] == msgValue[i2] else 0
                         
                #sc_list[i].isCorrPath = 1 if (self.trdata[pos_rev] == msgValue[i] and sc_list[i].isCorrPath == 1) else 0
                #copy_branch.isCorrPath = 1 if (self.trdata[pos_rev] == msgValue[i2] and copy_branch.isCorrPath == 1) else 0
                sc_list.append(copy_branch)
        else:
            self.edgeOrder = copy.deepcopy(PM_sorted_idx[:self.curr_list_size]) #self.edgeOrder is a pointer and any change will be reflected in PM_sorted_idx
            #Recognizing inactive paths:

                
            surviving_paths = np.zeros(2*self.curr_list_size, dtype=int) #the paths to be retianed among paths with indices L...2L-1
            surviving_paths[self.edgeOrder] = 1
            prunning_paths_indices = []

            for i in range(self.curr_list_size):
                if surviving_paths[i] == 0 and surviving_paths[i+self.curr_list_size] == 0:
                    prunning_paths_indices.append(i)

            for i in range(self.curr_list_size):
                if surviving_paths[i] == 1:
                    if surviving_paths[i+self.curr_list_size] == 1:    # Duplication needed
                        repl_idx = prunning_paths_indices[0]
                        prunning_paths_indices.pop(0)
                        i2 = i+self.curr_list_size
                        self.sc_list[repl_idx] = copy.deepcopy(self.sc_list[i])
                        self.sc_list[repl_idx].decoded[pos] = msgValue[i2]
                        self.sc_list[repl_idx].polar_decoded[pos] = edgeValue[i2]
                        self.sc_list[repl_idx].cur_state = pathState[i2]
                        self.sc_list[repl_idx].corrprob = pathMetric[i2]
                        self.sc_list[repl_idx].isCorrPath = 1 if (self.trdata[pos] == msgValue[i2] and self.sc_list[repl_idx].isCorrPath == 1) else 0
                        self.sc_list[repl_idx].pathOrder = i
                    self.sc_list[i].decoded[pos] = msgValue[i]
                    self.sc_list[i].polar_decoded[pos] = edgeValue[i]
                    self.sc_list[i].cur_state = pathState[i]
                    self.sc_list[i].corrprob = pathMetric[i]
                    self.sc_list[i].isCorrPath = 1 if (self.trdata[pos] == msgValue[i] and self.sc_list[i].isCorrPath == 1) else 0
                    self.sc_list[i].pathOrder = i
                elif surviving_paths[i] == 0:
                    if surviving_paths[i+self.curr_list_size] == 1:    # Swapping needed
                        i2 = i+self.curr_list_size
                        self.sc_list[i].decoded[pos] = msgValue[i2]
                        self.sc_list[i].polar_decoded[pos] = edgeValue[i2]
                        self.sc_list[i].cur_state = pathState[i2]
                        self.sc_list[i].corrprob = pathMetric[i2]
                        self.sc_list[i].isCorrPath = 1 if (self.trdata[pos] == msgValue[i2] and self.sc_list[i].isCorrPath == 1) else 0
                        self.sc_list[i].pathOrder = i
                        surviving_paths[i+self.curr_list_size] = 0
                    


    def pac_list_crc_decoder(self, soft_mess, issystematic, isCRCinc, crc1, L):
        #Successive cancellation list decoder"""

        # init list of decoding branches
        codeword_length = self.codeword_length
        log_N = self.log2_N

        self.sc_list = [path(codeword_length,self.m)]    #Branch is equivalent to one edge of the paths at each step on the binary tree, whihch carries intermediate LLRs, Partial sums, prob

        # initial/channel LLRs
        self.sc_list[0].llrs[codeword_length - 1:] = soft_mess
        self.list_size = L
        crc_len = crc1.len
        decoding_failed = False
        corr_path_is_found = 0
        elim_recorded = 0
        #elim_not_indicated = True
        for j in range(codeword_length):
            corr_path_not_found = 0
            i = pcfun.bitreversed(j, self.n)
            self.curr_list_size = len(self.sc_list)

            for l in self.sc_list:
                l.update_llrs(i)    #Update intermediate LLRs

            if self.polarcode_mask[i] == 1:
                self.pac_fork(self.sc_list, i)

            else:
                for l in self.sc_list:
                    edgeValue0 = pcfun.conv_1bit(0, l.cur_state, self.gen)
                    l.cur_state = pcfun.getNextState(0, l.cur_state, self.m)
                    cur_state0 = l.cur_state
                    l.decoded[i] = self.polarcode_mask[i]
                    l.polar_decoded[i] = edgeValue0

                    penalty = np.abs(l.llrs[0])
                    if l.llrs[0] < 0:
                        pathMetric0 = l.corrprob + (0 if edgeValue0==1 else 1) * penalty
                    else:
                        pathMetric0 = l.corrprob + (0 if edgeValue0==0 else 1) * penalty
                    l.corrprob = pathMetric0
            ii=0    # Counter for list elements
            corr_path_is_found = 0
            for l in self.sc_list:
                l.update_bits(i)

                
        if isCRCinc:
            self.sc_list.sort(key=lambda branch: branch.corrprob, reverse=False) #for prob-based: reverse=True #key: a function to specify the sorting criteria(s), reverse=True : in descending order
            if issystematic:
                self.mul_matrix(self.sc_list[0].decoded)
            best = self.extract(self.sc_list[0].decoded)
            if pcfun.np.sum(crc1.crcCalc(best)) == 0:

                self.repeat_no = -1
                self.shft_idx = 0
                return best[0:len(best)]
            else:
                idx=2
                for br in self.sc_list[1:]:
                    if issystematic:
                        self.mul_matrix(br.decoded)
                    rx = self.extract(br.decoded)
                    if pcfun.np.sum(crc1.crcCalc(rx)) == 0:
                        return rx[0:len(rx)]
                    idx+=1

            return best[0:len(best)]


        else:
            self.sc_list.sort(key=lambda branch: branch.corrprob, reverse=False)
            best = self.sc_list[0].decoded
            if issystematic:
                self.mul_matrix(best)
            return self.extract(best)

