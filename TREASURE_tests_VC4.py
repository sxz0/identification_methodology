import time
import os
import sys
import socket,fcntl,struct
from videocore.v3d import *
from videocore.driver import Driver
import random
import hashlib
import psutil

dri=Driver()

def sleep(duration):
    duration=duration*1000000000
    now = time.perf_counter_ns()
    end = now + duration
    while now < end:
        now = time.perf_counter_ns()

def get_QPU_freq(s):
    with RegisterMapping(dri) as regmap:
        with PerformanceCounter(regmap, [13,14,15,16,17,18,19]) as pctr:
            time.sleep(s)
            result = pctr.result()
            return sum(result)

def cpu_random():
    with RegisterMapping(dri) as regmap:
        with PerformanceCounter(regmap, [13,14,15,16,17,28,19]) as pctr:
            a=random.random()
            result = pctr.result()
            return (sum(result))

def cpu_true_random(n):
    with RegisterMapping(dri) as regmap:
        with PerformanceCounter(regmap, [13,14,15,16,17,28,19]) as pctr:
            a=os.urandom(n)
            result = pctr.result()
            return (sum(result))

def cpu_hash():
    with RegisterMapping(dri) as regmap:
         with PerformanceCounter(regmap, [13,14,15,16,17,28,19]) as pctr:
             h=int(hashlib.sha256("test string".encode('utf-8')).hexdigest(), 16) % 10**8
             result = pctr.result()
             return (sum(result))

def getHwAddr(ifname):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        info = fcntl.ioctl(s.fileno(), 0x8927,  struct.pack('256s', bytes(ifname, 'utf-8')[:15]))
        return ':'.join('%02x' % b for b in info[18:24])
    except:
        return "00:00:00:00:00:00"
    
def main():
	s=int(sys.argv[1])
	r=int(sys.argv[2])
	
	results=[]
	results.append(os.popen("vcgencmd measure_temp | cut -d = -f 2 | cut -d \"'\" -f 1").read()[:-1])
	results.append(get_QPU_freq(s))

	results.append(0)
	results.append(0)
	
	results.append(cpu_hash())
	results.append(cpu_random())
	results.append(cpu_true_random(r))
	
	inter=list(psutil.net_if_addrs().keys())
	inter.remove('lo')
	inter=inter[0]
	mac=getHwAddr(inter)
	results.append(mac)
	
	print(*results, sep=',')

if __name__ == "__main__":
    main()

