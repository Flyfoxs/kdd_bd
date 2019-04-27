import pandas as pd
from glob import glob

def vote():
    for file in glob('./output/ensemble/vote/*.csv'):

        print(file)


if __name__ == '__main__':
    vote()