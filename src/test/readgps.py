import serial
from time import time

with serial.Serial('COM6', 230400) as ser:
    t = time()
    while True:
        l = ser.readline()
        print(l)
        # if b"$GNGSA" in l:
        #     print(l.decode().split(',')[-4:-1])
        if b"$GNRMC" in l:  # 緯度経度方向の情報を取り出す
            print(time()-t)
            t = time()
