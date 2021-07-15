import os


def get_src_directory():
    '''Return absolute path to src directory'''
    pwd = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")  + "/"
    return pwd