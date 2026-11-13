# Visual Basic's Rnd generator, which the waveform's tables are drawn from.
import struct, numpy as np
MASK=0xFFFFFF
class VbRnd:
    def __init__(s): s.x=0x050000
    def adv(s): s.x=(s.x*0xFD43FD+0xC39EC3)&MASK
    def val(s): return np.float32(s.x)/np.float32(16777216.0)
    def rnd_neg1(s):                               # Rnd(-1)
        b=struct.unpack('<I',struct.pack('<f',-1.0))[0]; s.x=(b+(b>>24))&MASK; s.adv()
    def randomize(s,d):
        hi=struct.unpack('<Q',struct.pack('<d',float(d)))[0]>>32; mid=((hi&0xFFFF)^(hi>>16))&0xFFFF; s.x=(s.x&0xFF0000FF&MASK)|(mid<<8)
    @classmethod
    def stream(c,seed):
        r=c(); r.rnd_neg1(); r.randomize(seed); return r
    def draw(s,m): s.adv(); return int(np.float32(s.val()*np.float32(m)))
    def rand_int(s,n): return s.draw(n+1)
def phase_indices(seed=43.0,bins=128,rows=201):
    r=VbRnd.stream(seed); return np.array([[r.rand_int(32) for _ in range(rows)] for _ in range(bins)])
