# OFDM levels 10-16 of the 2300 Hz mode (49 carriers from bin 8, 46.875 Hz apart, 1024 + 128 samples per symbol).
# Levels 10-14: differential 4-/8-phase with one marker per carrier. Levels 15-16: coherent 16/32QAM with pilot cells.
# Own implementation of the published rules; checked by round trip only (no public recording contains these levels).
import sys, numpy as np
from .vb import VbRnd
from . import turbo, hffsk, hfpsk
# level: symbols per frame, bits per cell, puncture period, info bits
LEV={10:(150,2,2,7197),11:(150,2,4,9596),12:(150,2,8,11512),13:(150,3,4,14398),14:(150,3,8,17276),15:(155,4,10,21710),16:(155,5,8,26056)}
NC,FIRST,MK,PP=49,8,1,8
h=np.sqrt(0.5); Q16=[-0.7778175,-0.2592725,0.7778175,0.2592725]; Q32=[0.2121320,0.6363961,1.0606602]
_b=[complex(Q32[i],Q32[q]) for i,q in ((0,0),(1,0),(1,2),(2,0),(0,1),(1,1),(0,2),(2,1))]
CST={2:np.array([1,1j,-1j,-1]),3:np.array([1,h+1j*h,-h+1j*h,1j,h-1j*h,-1j,-1,-h-1j*h]),
     4:np.array([complex(Q16[v>>2],Q16[v&3]) for v in range(16)]),5:np.array(_b+[z.conjugate() for z in _b]+[-z.conjugate() for z in _b]+[-z for z in _b])}
def pilot(b,s): return s%PP==(2 if b%2==0 else 3)
def n_of(spf,bps): return NC*(spf-2-MK)*bps if bps<4 else NC*bps*(spf-1-MK-((spf-3)//PP+1))
def tables(upto=16):
    r=hfpsk.tables()['rng']; T={}
    for lv in range(10,upto+1):
        spf,bps,p,k=LEV[lv]; chan=hffsk.perm(r,n_of(spf,bps)); tp=hffsk.perm(r,k); mark=np.zeros((NC,spf-1),bool)
        for c in range(NC):
            v=3+(FIRST+c)%2+PP*r.draw((spf-3)//PP) if bps>=4 else 5+r.draw(spf-8); mark[c,v-2]=True
        T[lv]=(chan,tp,mark)
    return T
def phasors():
    r=VbRnd.stream(292.0); d=[r.rand_int(128) for _ in range(65000)]; out=np.zeros((128,469),int); it=iter(d)
    for a,b in ((0,409),(409,469)):
        for row in range(128):
            for m in range(a,b): out[row,m]=next(it)
    return np.exp(2j*np.pi*out/128)
def cells(lv,T,tx=None,rx=None):
    """walk the frame grid in transmission order. With tx: return rows (symbols x carriers). With rx: return the list of (row, carrier, kind)."""
    spf,bps,p,k=LEV[lv]; mark=T[lv][2]; qam=bps>=4; cst=CST[bps]; rows=np.zeros((spf-1,NC),complex); order=[]; ptr=0
    for bi in range(NC):
        b=FIRST+bi; cell=1.0+0j
        for s in range(2,spf+1):
            m=mark[bi,s-2]; kind='ref'
            if qam:
                if pilot(b,s): cell=1.0+0j; kind='pilot'
                elif m: cell=cell/abs(cell) if abs(cell)>0 else 1.0+0j; kind='marker'
                else:
                    kind='data'
                    if tx is not None: v=int(''.join(map(str,tx[ptr:ptr+bps])),2); ptr+=bps; cell=cst[v]
            elif s>2:
                if m: cell=cell/abs(cell); kind='marker'
                else:
                    kind='data'
                    if tx is not None: v=int(''.join(map(str,tx[ptr:ptr+bps])),2); ptr+=bps; cell=cell*cst[v]
            rows[s-2,bi]=cell; order.append((s-2,bi,kind))
    return rows,order
def make(lv,frame,T,P,noise=0.0,seed=1,df=0.0):
    spf,bps,p,k=LEV[lv]; chan,tp,mark=T[lv]; rng=np.random.default_rng(seed)
    bits=[(b>>i)&1 for b in frame for i in range(7,-1,-1)]; bits+=[0]*(k-len(bits)); r=VbRnd.stream(292.0); bits=[b^r.rand_int(1) for b in bits]
    coded=turbo.encode(bits,tp,p,len(chan)); tx=[coded[i] for i in chan]; rows,_=cells(lv,T,tx=tx)
    rows=rows*np.conj(P[FIRST:FIRST+NC,:spf-1]).T; lead=4 if bps<4 else 5; allrows=np.vstack([CST[bps][rng.integers(0,len(CST[bps]),(lead,NC))],rows]); out=[]
    for row in allrows:
        buf=np.zeros(1024,complex); buf[FIRST:FIRST+NC]=row; x=2*np.real(np.fft.ifft(buf)*1024); out.append(np.concatenate([x[-128:],x]))
    x=np.concatenate([np.zeros(4800)]+out+[np.zeros(4800)])
    if df: x=np.real((x+1j*np.imag(_hil(x)))*np.exp(2j*np.pi*df*np.arange(len(x))/48000))
    return (x/np.abs(x).max()+rng.normal(0,noise,len(x)))*8000,lead
def _hil(x):
    F=np.fft.fft(x); n=len(x); hh=np.zeros(n); hh[0]=1; hh[1:(n+1)//2]=2
    if n%2==0: hh[n//2]=1
    return np.fft.ifft(F*hh)
def receive(x,lv,T,P,verbose=True,score_only=False):
    spf,bps,p,k=LEV[lv]; chan,tp,mark=T[lv]; qam=bps>=4; cst=CST[bps]; nrows=spf-1; z=_hil(x)
    # symbol timing and tuning offset from the guard interval (each symbol's last 128 samples repeat at its start)
    c=z[:-1024]*np.conj(z[1024:]); e=np.abs(z[:-1024])**2+np.abs(z[1024:])**2; w=np.ones(128); cs=np.convolve(c,w,'valid'); es=np.convolve(e,w,'valid')+1e-9
    n=len(cs)//1152*1152; fold=cs[:n].reshape(-1,1152).sum(0); ph=int(np.argmax(np.abs(fold))); df=-np.angle(fold[ph])/(2*np.pi*1024/48000)
    zz=z*np.exp(-2j*np.pi*df*np.arange(len(z))/48000); nsym=(len(zz)-ph)//1152
    G=np.array([np.fft.fft(zz[ph+q*1152+96:ph+q*1152+1120])[FIRST:FIRST+NC] for q in range(nsym)])*np.exp(2j*np.pi*(FIRST+np.arange(NC))*32/1024)
    _,order=cells(lv,T); best=None
    for q0 in range(0,nsym-nrows+1):                       # which symbol is row 0 of the frame
        D=G[q0:q0+nrows]*P[FIRST:FIRST+NC,:nrows].T
        if qam: v=np.array([D[r,c] for r,c,kd in order if kd=='pilot']).reshape(NC,-1); sc=np.abs(v.sum(1)).sum()/(np.abs(v).sum()+1e-12)
        else: v=np.array([D[r,c]*np.conj(D[r-1,c]) for r,c,kd in order if kd=='marker']); sc=abs(v.sum())/(np.abs(v).sum()+1e-12)
        if best is None or sc>best[0]: best=(sc,q0)
    if best is None: return (False,b'',0.0,0.0)
    score,q0=best
    if score_only: return (False,b'',score,df)
    D=G[q0:q0+nrows]*P[FIRST:FIRST+NC,:nrows].T; llr=[]
    bitmap=np.array([[(v>>(bps-1-j))&1 for j in range(bps)] for v in range(1<<bps)])
    def soft(y,sig2):
        d2=np.abs(y-cst)**2/sig2; return [d2[bitmap[:,j]==1].min()-d2[bitmap[:,j]==0].min() for j in range(bps)]
    if qam:
        H=np.zeros((nrows,NC),complex)
        for c in range(NC):
            pr=np.array([r for r,cc,kd in order if cc==c and kd=='pilot']); hv=D[pr,c]; rr=np.arange(nrows); H[:,c]=np.interp(rr,pr,hv.real)+1j*np.interp(rr,pr,hv.imag)
        E=D/H; sig2=0.02
        for r,c,kd in order:
            if kd=='data': llr+=soft(E[r,c],sig2)
    else:
        for r,c,kd in order:
            if kd=='data':
                d=D[r,c]*np.conj(D[r-1,c]); d=d/(abs(D[r-1,c])**2+1e-12); llr+=soft(d,0.1)
    llr=np.clip(np.array(llr),-30,30); coded=np.zeros(len(chan)); coded[np.array(chan)]=llr
    bits=turbo.decode(coded,tp,p,k,iters=6); r=VbRnd.stream(292.0); bits=[int(b)^r.rand_int(1) for b in bits]
    fr=bytes(int(''.join(map(str,bits[i:i+8])),2) for i in range(0,k//8*8,8)); ok=hffsk.crc16(fr[:-2])==int.from_bytes(fr[-2:],'big')
    if verbose: print('level %d: frame score %.2f, row 0 at symbol %d, tuning offset %+.1f Hz, %d-byte frame, CRC %s'%(lv,score,q0,df,len(fr),'OK' if ok else 'wrong'))
    return ok,fr,score,df
