import os
path = os.path.split(os.path.realpath(__file__))[0]
command = r'setx WORK1 %s /m' % path
os.system(command)
