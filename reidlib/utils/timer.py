import time
from tqdm import tqdm

def wait(h=0, m=0):
    mins = 60*h + m
    for i in tqdm(range(mins), desc='waiting for {} mins'.format(mins)):
        time.sleep(60)