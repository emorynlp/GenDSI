

from dst.data.dst_data import DstData
import os

print(DstData)
print('File running', __file__)
print('Working Directory', os.getcwd())
print('Python path', os.environ['PYTHONPATH'].split(os.pathsep))