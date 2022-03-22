# List Decoder for Polar Codes, CRC-Polar Codes, and PAC Codes
If you find this algorithm useful, please cite the following paper. Thanks.

M. Rowshan, A. Burg and E. Viterbo, "Polarization-Adjusted Convolutional (PAC) Codes: Sequential Decoding vs List Decoding," in IEEE Transactions on Vehicular Technology, vol. 70, no. 2, pp. 1434-1447, Feb. 2021, doi: 10.1109/TVT.2021.3052550.

https://ieeexplore.ieee.org/abstract/document/9328621

Description: 
This is an implementation of the successive cancellation list (SCL) decoding algorithm for polar codes, CRC-polar codes, and PAC codes with the choice of various code constructions/rate-profiles. 
The list decoding algorithm is an adaptive two stage successive cancellation list (SCL) algorithm. That means first it tries L=1 and then L=L_max. The performance is the same as list decoding with L_max. This trick has been implemented in the simulator.py file. The rest of the files are the same as the standard list decoding algorithm.

To switch between decoding polar codes and PAC codes, you need to change the generator polynomial conv_gen to conv_gen=[1] for polar codes or any other polynomial such as conv_gen=[1,0,1,1,0,1,1].

Note that the "copy on write" or "lazy copy" technique has been used in this algorithm.
