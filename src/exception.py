import sys 
from src.logger import logging
def error_message_details(error,error_datail:sys):
    _,_,exc_tb=error_datail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="error occured in python scrpit name[{0}] \
    in line number [{1}] error message[{2}] ".format(file_name,exc_tb.tb_lineno,str(error))
    return error_message

class CustomException(Exception):
    def __init__(self, error_message,error_details:sys):    
        super().__init__(error_message)
        self.error_message=error_message_details(error_message,error_datail=error_details)
    def __str__(self):
        return self.error_message

