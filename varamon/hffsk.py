# tone-keyed (FSK) frames of the 2300 Hz mode, levels 1-3. Own implementation of the published rules.
# Finds a frame by its 24 pilot symbols (search over timing and tuning offset), soft-demodulates, turbo-decodes, checks the CRC.
import sys, wave, numpy as np
from .vb import VbRnd
from . import turbo
# level: symbols, bits/symbol, puncture period, sps, first bin, last bin, step, info bits        (bin width = 3000/sps Hz)
LEV={1:(125,3,1,128,33,96,2,96),2:(229,3,1,64,17,48,2,200),3:(229,4,2,64,17,48,2,402),4:(396,4,2,32,9,24,1,736)}
LEVS={2300:dict(LEV),
      500:{1:(125,3,1,128,54,74,2,96),2:(229,3,1,64,27,37,1,200),3:(227,3,2,64,27,37,1,297)},
      2750:{1:(125,3,1,128,24,103,2,96),2:(229,3,1,64,12,51,2,200),3:(229,4,2,64,12,51,2,402),4:(396,4,2,32,6,25,1,736)}}
def set_bw(bw):
    """switch the level table to another bandwidth mode (500, 2300 or 2750 Hz); each mode draws its own tables"""
    LEV.clear(); LEV.update(LEVS[bw])
GRAY=[0,1,3,2,7,6,4,5,15,14,12,13,8,9,11,10]
def perm(r,n):
    used=[False]*n; out=[]
    while len(out)<n:
        v=r.draw(n)
        if not used[v]: used[v]=True; out.append(v)
    return out
def tables():
    r=VbRnd.stream(43.0); T={}
    for lv in sorted(LEV):
        s,bps,p,sps,first,last,step,k=LEV[lv]; n=(s-1-24)*bps; chan=perm(r,n); tp=perm(r,k); nb=last-first+1; m=nb//step+1 if step>1 else nb
        hop=[first+(1 if (step>1 and i%2==0) else 0)+step*r.draw(m) for i in range(s-1)]
        inc=np.float32(s-2)/np.float32(24); acc=np.float32(1.0); rows=[]
        for _ in range(24): acc=np.float32(acc+inc); rows.append(int(acc))
        scale=(1<<bps)//8; pilot=[None]*(s-1)
        for row in rows: pilot[row-2]=scale*r.draw(8)
        T[lv]=(chan,tp,hop,pilot)
    T['rng']=r                                   # generator state after level 4, for the levels that follow
    return T
def crc16(d):
    c=0xFFFF
    for b in d:
        c^=b<<8
        for _ in range(8): c=((c<<1)^0x1021)&0xFFFF if c&0x8000 else (c<<1)&0xFFFF
    return c^0xFFFF
def load48(fn):
    w=wave.open(fn); fs=w.getframerate()
    if w.getsampwidth()!=2: raise ValueError('16-bit PCM WAV expected')
    x=np.frombuffer(w.readframes(w.getnframes()),'<i2').reshape(-1,w.getnchannels())[:,0].astype(float)
    if fs!=48000:
        t=np.arange(int(len(x)*48000/fs))*(fs/48000); i=np.floor(t).astype(int); k=np.arange(-7,9); out=np.empty(len(t))
        for a in range(0,len(t),1<<17):
            u=(t[a:a+(1<<17)]-i[a:a+(1<<17)])[:,None]-k; wgt=np.sinc(u)*(0.5+0.5*np.cos(np.pi*u/8)); out[a:a+(1<<17)]=(x[np.clip(i[a:a+(1<<17),None]+k,0,len(x)-1)]*wgt).sum(1)
        x=out
    return x
def try_level(x,lv,T,t_lo,t_hi,verbose=True):
    s,bps,p,sps,first,last,step,k=LEV[lv]; chan,tp,hop,pilot=T[lv]; N=16*sps; nb=last-first+1; nsym=s-1
    F=np.fft.fft(x); f=np.fft.fftfreq(len(x),1/48000); F[f<0]=0; z=np.fft.ifft(2*F); tt=np.arange(len(z))/48000
    def binof(i,t): return first+(hop[i]+step*t-first)%nb
    pil=[(i,binof(i,v)) for i,v in enumerate(pilot) if v is not None]; best=None
    for df in np.arange(-90,91,6.0):
        zz=z*np.exp(-2j*np.pi*df*tt); hopS=N//8; nfr=(len(zz)-N)//hopS
        S=np.abs(np.array([np.fft.fft(zz[q*hopS:q*hopS+N])[:first+nb+2] for q in range(int(t_lo*48000/hopS),min(nfr,int(t_hi*48000/hopS)+nsym*8+8))]))**2
        base=int(t_lo*48000/hopS)
        for q0 in range(0,int((t_hi-t_lo)*48000/hopS)):
            idx=[q0+i*8 for i,_ in pil]
            if idx[-1]>=len(S): break
            num=sum(S[a,b] for a,(_,b) in zip(idx,pil)); den=sum(S[a,first:first+nb].sum() for a in idx)+1e-9
            if best is None or num/den>best[0]: best=(num/den,df,(base+q0)*hopS)
    score,df,start=best
    zz=z*np.exp(-2j*np.pi*df*tt); llr=[]
    for i in range(nsym):
        if pilot[i] is not None: continue
        X=np.abs(np.fft.fft(zz[start+i*N:start+(i+1)*N]))**2; e=np.array([X[binof(i,GRAY[v])] for v in range(1<<bps)]); e=e/(e.mean()+1e-12)
        for b in range(bps-1,-1,-1):
            m=np.array([(v>>b)&1 for v in range(1<<bps)]); llr.append(np.log(e[m==0].max()+1e-9)-np.log(e[m==1].max()+1e-9))
    llr=np.array(llr); coded=np.zeros(len(chan)); coded[np.array(chan)]=llr
    bits=turbo.decode(coded*2,tp,p,k,iters=10); r=VbRnd.stream(292.0); bits=[int(b)^r.rand_int(1) for b in bits]
    fr=bytes(int(''.join(map(str,bits[i:i+8])),2) for i in range(0,k//8*8,8)); ok=crc16(fr[:-2])==int.from_bytes(fr[-2:],'big')
    if verbose: print('level %d: pilot score %.2f at %.3f s, tuning offset %+.0f Hz -> %d-byte frame %s  CRC %s'%(lv,score,start/48000,df,len(fr),fr.hex(),'OK' if ok else 'wrong'))
    return ok,fr,score
def make(lv,frame,T,noise=0.0,seed=1,df=0.0):
    """audio (48 kHz) of one frame: lead-in symbols, then the frame"""
    s,bps,p,sps,first,last,step,k=LEV[lv]; chan,tp,hop,pilot=T[lv]; N=16*sps; nb=last-first+1; rng=np.random.default_rng(seed)
    bits=[(b>>i)&1 for b in frame for i in range(7,-1,-1)]; bits+= [0]*(k-len(bits)); r=VbRnd.stream(292.0); bits=[b^r.rand_int(1) for b in bits]
    coded=turbo.encode(bits,tp,p,len(chan)); tx=[coded[i] for i in chan]; ptr=0; bins=[]
    lead=int(np.ceil(0.080/(sps/3000)-1e-9)); bins=[first+(hop[i+1]+step*int(rng.integers(0,1<<bps))-first)%nb for i in range(lead)]
    for i in range(s-1):
        if pilot[i] is not None: t=pilot[i]
        else: v=int(''.join(map(str,tx[ptr:ptr+bps])),2); ptr+=bps; t=GRAY[v]
        bins.append(first+(hop[i]+step*t-first)%nb)
    n=np.arange(N); x=np.concatenate([np.sin(2*np.pi*(b*48000/N+df)*n/48000) for b in bins]); x=np.concatenate([np.zeros(9600),x,np.zeros(9600)])
    return (x+rng.normal(0,noise,len(x)))*8000
