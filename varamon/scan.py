# Scanner: finds transmissions in a recording, works out the signal type and level of each, decodes it and checks the CRC.
# 2300 Hz mode, levels 1-16. Receive only.
import sys, os, pickle, time, wave, numpy as np
from . import hffsk, hfpsk, hfofdm
CACHE=os.path.join(os.path.expanduser('~'),'.varamon','tables-v2.pkl')
def tables():
    """all generated tables of the 2300 Hz mode (takes a minute the first time, then cached)"""
    if os.path.exists(CACHE): return pickle.load(open(CACHE,'rb'))
    Tf=hffsk.tables(); Tf.pop('rng',None); Tp=hfpsk.tables(); Tp.pop('rng',None); To=hfofdm.tables(); other={}
    for bw in (500,2750):                      # tone-keyed levels of the other two bandwidth modes
        hffsk.set_bw(bw); t=hffsk.tables(); t.pop('rng',None); other[bw]=t
    hffsk.set_bw(2300)
    out=dict(fsk=Tf,fsk500=other[500],fsk2750=other[2750],psk=Tp,ofdm=To,pp=hfpsk.phasors(),po=hfofdm.phasors()); os.makedirs(os.path.dirname(CACHE),exist_ok=True); pickle.dump(out,open(CACHE,'wb')); return out
def segments(x):
    F=np.fft.rfft(x); f=np.fft.rfftfreq(len(x),1/48000); F[(f<300)|(f>2900)]=0; y=np.fft.irfft(F,len(x)); blk=480
    e=np.sqrt((y[:len(y)//blk*blk].reshape(-1,blk)**2).mean(1)); thr=max(np.percentile(e,10)*3,e.max()*0.08); on=e>thr
    ed=np.flatnonzero(np.diff(np.r_[0,on.astype(int),0])); seg=[[a,b] for a,b in zip(ed[::2],ed[1::2])]; out=[]
    for a,b in seg:
        if out and a-out[-1][1]<45: out[-1][1]=b                       # bridge fades shorter than 0.45 s
        else: out.append([a,b])
    return [(a*blk,b*blk) for a,b in out if (b-a)*blk>=3.0*48000],y
def kind_of(s):
    """'fsk' (one tone at a time), 'ofdm' (guard interval repeats) or 'psk'"""
    share=0.0
    for N,lo,hi in ((1024,6,64),(512,3,32)):               # the fastest tone level has symbols of only 512 samples
        fr=np.array([np.abs(np.fft.rfft(s[i:i+N]*np.hanning(N)))[lo:hi]**2 for i in range(0,len(s)-N,N)]); share=max(share,float(np.median(fr.max(1)/(fr.sum(1)+1e-9))))
    if share>0.3: return 'fsk'
    z=hfofdm._hil(s); c=z[:-1024]*np.conj(z[1024:]); e=np.abs(z[:-1024])**2; w=np.ones(128); cs=np.convolve(c,w,'valid'); n=len(cs)//1152*1152
    g=np.abs(cs[:n].reshape(-1,1152).sum(0)).max()/(e.sum()*128/1152+1e-9)
    return 'ofdm' if g>0.35 else 'psk'
def decode_segment(s,T):
    k=kind_of(s); res=None
    if k=='fsk':
        try:
            for bw,key in ((2300,'fsk'),(2750,'fsk2750'),(500,'fsk500')):
                hffsk.set_bw(bw)
                for lv in sorted(hffsk.LEV):
                    flen=(hffsk.LEV[lv][0]-1)*16*hffsk.LEV[lv][3]/48000
                    if len(s)/48000<flen-0.2: continue
                    ok,fr,score=hffsk.try_level(s,lv,T[key],0.0,min(2.9,max(0.05,len(s)/48000-flen)),verbose=False)
                    if ok: return k,lv,fr,True,score,bw
                    if res is None or score>res[4]: res=(k,lv,fr,False,score,bw)
        finally: hffsk.set_bw(2300)
    elif k=='psk':
        sc=[(hfpsk.receive(s,lv,T['psk'],T['pp'],0.0,2.9,verbose=False,score_only=True)[2],lv) for lv in (5,6,7,8,9)]; score,lv=max(sc)
        ok,fr,score=hfpsk.receive(s,lv,T['psk'],T['pp'],0.0,2.9,verbose=False); res=(k,lv,fr,ok,score,2300)
    else:
        sc=[(hfofdm.receive(s,lv,T['ofdm'],T['po'],verbose=False,score_only=True)[2],lv) for lv in (10,11,12,13,14,15)]; sc.sort(reverse=True)
        order=[sc[0][1]]+([16] if sc[0][1]==15 else [])
        for lv in order:
            ok,fr,score,df=hfofdm.receive(s,lv,T['ofdm'],T['po'],verbose=False); res=(k,lv,fr,ok,score,2300)
            if ok: break
    return res
def scan(x,T,report=print):
    segs,y=segments(x); out=[]
    for a,b in segs:
        s=x[max(0,a-96000):b+33600]; t0=time.time(); r=decode_segment(s,T)        # 2 s margin before, 0.7 s after: a faded start or end is common off air
        if r is None: continue
        k,lv,fr,ok,score,bw=r; c=fr[-3] if len(fr)>=3 else 0
        out.append(dict(t=a/48000,dur=(b-a)/48000,kind=k,level=lv,ok=ok,frame=fr,score=score,bw=bw))
        report('%7.2f s  %5.2f s  %-4s %4d Hz level %2d  %4d bytes  CRC %-5s  control %02x (reply %d, compressed %d, last %d, full %d)  score %.2f  [%.0f s]'%(a/48000,(b-a)/48000,k,bw,lv,len(fr),'OK' if ok else 'wrong',c,c>>6,(c>>5)&1,(c>>1)&1,c&1,score,time.time()-t0))
    return out

def mix(seed=11):
    """test audio with generated frames: 2300 Hz levels 2, 6, 11, 15, then 500 Hz level 2 and 2750 Hz level 4.
    Returns (samples at 48 kHz, list of (bandwidth, level, frame))"""
    T=tables(); rng=np.random.default_rng(seed); parts=[np.zeros(24000)]; sent=[]
    def body(k):
        b=bytes(rng.integers(0,256,k//8-2).tolist()); return b+hffsk.crc16(b).to_bytes(2,'big')
    for lv in (2,6,11,15):
        k=hffsk.LEV[lv][7] if lv<=4 else hfpsk.LEV[lv][4] if lv<=9 else hfofdm.LEV[lv][3]; fr=body(k); sent.append((2300,lv,fr))
        if lv<=4: xx=hffsk.make(lv,fr,T['fsk'],noise=0.02,df=4.0)
        elif lv<=9: xx=hfpsk.make(lv,fr,T['psk'],T['pp'],noise=0.02,df=4.0)
        else: xx=hfofdm.make(lv,fr,T['ofdm'],T['po'],noise=0.02,df=4.0)[0]
        parts+=[xx/np.abs(xx).max()*8000,np.zeros(36000)]
    try:
        for bw,key,lv in ((500,'fsk500',2),(2750,'fsk2750',4)):
            hffsk.set_bw(bw); fr=body(hffsk.LEV[lv][7]); sent.append((bw,lv,fr)); xx=hffsk.make(lv,fr,T[key],noise=0.02,df=4.0); parts+=[xx/np.abs(xx).max()*8000,np.zeros(36000)]
    finally: hffsk.set_bw(2300)
    return np.concatenate(parts),sent
