class FatalException(Exception):
    pass
    
class ErrorRestartException(Exception):
    pass

class InfoRestartException(Exception):
    pass

class ResetDSPException(Exception):
    pass

class NextMsgException(Exception):
    pass
    
class TimeoutException(Exception):
    pass
    
class InvalidParamterException(Exception):
    ''' Raised when invalid parameters are detected in the
        parameter server this should be considered a fatal exception
    '''
    pass
