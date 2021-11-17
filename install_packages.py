#this code install line by line a list of pip package 
import sys
from pip._internal import main as pip_main

error_log = open('error_log.txt','w')

def install(package):
    try:
        pip_main(['install', package])
    except Exception as e:
        error_log.write(str(e))

if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        for line in f:
            install(line)
    error_log.close()
