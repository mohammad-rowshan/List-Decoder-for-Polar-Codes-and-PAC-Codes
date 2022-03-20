class PCLengthError(Exception):
    """An exception for cases when information_size K
    bigger than codeword length N"""
    pass


class PCLengthDivTwoError(Exception):
    """An exception for cases if codeword length N
     is not divisible by 2"""
    pass


class PCInfoLengthError(Exception):
    """Check if length of information message is equal to
    information_size of the PolarCode object"""