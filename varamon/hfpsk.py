# phase-keyed levels 5-9 of the 2300 Hz mode (3-10 carriers, 93.75 symbols/s, differential 4-phase).
# Own implementation of the published rules: generator and receiver. No public recording contains these levels, so this is
# checked by round trip only (own generator -> own receiver).
import sys, numpy as np
from .vb import VbRnd
from . import turbo, hffsk
# level: symbols per frame, markers per carrier, carrier centre frequencies (Hz), pulse, info bits
LEV={5:(376,12,(900,1500,2100),'rc',1080),6:(365,9,(750,1250,1750,2250),'rc',1410),7:(355,6,(700,1020,1340,1660,1980,2300),'rrc2',2076),
     8:(352,5,(702,930,1158,1386,1614,1842,2070,2298),'rrc2',2754),9:(349,4,(555,765,975,1185,1395,1605,1815,2025,2235,2445),'rrc1',3424)}
T48=512                                  # samples per symbol at 48 kHz
CST=np.array([1,1j,-1j,-1])
def tables():
    F=hffsk.tables(); r=F['rng']; T={}
    for lv in (5,6,7,8,9):
        spf,mk,fc,pulse,k=LEV[lv]; nc=len(fc); n=nc*(spf-2-mk)*2; chan=hffsk.perm(r,n); tp=hffsk.perm(r,k); mark=np.zeros((nc,spf-1),bool)
        for c in range(nc):
            row=[]
            while len(row)<mk:
                v=5+r.draw(spf-8)
                if v not in row: row.append(v)
            for v in row: mark[c,v-2]=True
        T[lv]=(chan,tp,mark)
    T['rng']=r                                   # generator state after level 9
    return T
def phasors():
    r=VbRnd.stream(292.0); d=np.zeros((13,471),int)
    for a,b in ((0,411),(411,471)):
        for c in range(13):
            for k in range(a,b): d[c,k]=r.rand_int(128)
    return np.exp(2j*np.pi*d/128)
def pulse(kind):
    if kind=='rc':
        t=np.arange(-256,257); u=t/256.0; h=np.ones(len(u))
        with np.errstate(divide='ignore',invalid='ignore'): h=np.sin(np.pi*u)*np.cos(np.pi*u)/(np.pi*u*(1-4*u*u))
        h[u==0]=1.0; h[np.abs(u)==0.5]=0.5*np.sin(np.pi/2)
    else:
        span=int(kind[3]); t=np.arange(-512*span,512*span+1); x=t/512.0
        with np.errstate(divide='ignore',invalid='ignore'): h=4*np.cos(2*np.pi*x)/(np.pi*(1-16*x*x))
        h[np.abs(x)==0.25]=1/np.sqrt(2)*((1+2/np.pi)*np.sin(np.pi/4)+(1-2/np.pi)*np.cos(np.pi/4))
    return h
def stagger(c,nc): return ([0,11,21][c] if nc==3 else (16 if c%2 else 0))*16
def make(lv,frame,T,P,noise=0.0,seed=1,df=0.0):
    spf,mk,fc,kind,k=LEV[lv]; chan,tp,mark=T[lv]; nc=len(fc); nsym=spf-1; rng=np.random.default_rng(seed)
    bits=[(b>>i)&1 for b in frame for i in range(7,-1,-1)]; bits+=[0]*(k-len(bits)); r=VbRnd.stream(292.0); bits=[b^r.rand_int(1) for b in bits]
    coded=turbo.encode(bits,tp,2,len(chan)); tx=[coded[i] for i in chan]; ptr=0; h=pulse(kind); lead=8; L=(lead+nsym+4)*T48+len(h)+400; out=np.zeros(L+9600)
    t=np.arange(len(out))/48000.0
    for c,f in enumerate(fc):
        syms=[CST[rng.integers(0,4)]*np.conj(P[c][q]) for q in range(lead)]; d=CST[rng.integers(0,4)]; syms.append(d*np.conj(P[c][0]))
        for s in range(1,nsym):
            if not (s>=2 and mark[c][s-2]): d=d*CST[2*tx[ptr]+tx[ptr+1]]; ptr+=2
            syms.append(d*np.conj(P[c][s]))
        b=np.zeros(len(out),complex); base=4800+stagger(c,nc)
        for i,v in enumerate(syms): b[base+i*T48:base+i*T48+len(h)]+=v*h
        out+=np.real(b*np.exp(2j*np.pi*(f+df)*t))
    assert ptr==len(tx)
    return (out/np.abs(out).max()+rng.normal(0,noise,len(out)))*8000
def receive(x,lv,T,P,t_lo,t_hi,verbose=True,score_only=False):
    spf,mk,fc,kind,k=LEV[lv]; chan,tp,mark=T[lv]; nc=len(fc); nsym=spf-1; h=pulse(kind); h=h/np.sum(h*h); t=np.arange(len(x))/48000.0
    Y=[np.convolve(x*np.exp(-2j*np.pi*f*t),h[::-1],'same') for f in fc]                 # matched filter per carrier
    best=None
    for t0 in range(int(t_lo*48000),int(t_hi*48000),16):
        acc=0; tot=0
        for c in range(nc):
            idx=t0+stagger(c,nc)+np.arange(nsym)*T48
            if idx[-1]>=len(x): break
            r=Y[c][idx]*P[c][:nsym]; d=r[1:]*np.conj(r[:-1]); m=np.flatnonzero(mark[c][:nsym-2])+2      # marker at symbol s: no phase change from s-1
            acc+=d[m-1].sum(); tot+=np.abs(d[m-1]).sum()
        else:
            sc=abs(acc)/(tot+1e-12)
            if best is None or sc>best[0]: best=(sc,t0,np.angle(acc))
    if best is None: return (False,b'',0.0)
    score,t0,theta=best; llr=[]
    if score_only: return (False,b'',score)
    for c in range(nc):
        idx=t0+stagger(c,nc)+np.arange(nsym)*T48; r=Y[c][idx]*P[c][:nsym]; d=r[1:]*np.conj(r[:-1])*np.exp(-1j*theta); d=d/(np.abs(d).mean()+1e-12)
        for s in range(1,nsym):
            if s>=2 and mark[c][s-2]: continue
            v=d[s-1]; llr+=[v.real+v.imag,v.real-v.imag]
    llr=np.array(llr); coded=np.zeros(len(chan)); coded[np.array(chan)]=llr
    bits=turbo.decode(coded*3,tp,2,k,iters=8); r=VbRnd.stream(292.0); bits=[int(b)^r.rand_int(1) for b in bits]
    fr=bytes(int(''.join(map(str,bits[i:i+8])),2) for i in range(0,k//8*8,8)); ok=hffsk.crc16(fr[:-2])==int.from_bytes(fr[-2:],'big')
    if verbose: print('level %d: marker score %.2f at %.3f s, tuning offset %+.1f Hz, %d-byte frame, CRC %s'%(lv,score,t0/48000,theta/(2*np.pi)*93.75,len(fr),'OK' if ok else 'wrong'))
    return ok,fr,score
