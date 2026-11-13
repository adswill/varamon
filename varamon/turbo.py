# the turbo code. Two identical 8-state recursive encoders (feedback u^r0^r1, parity a^r2^r0), the second fed through a
# permutation; systematic bits always sent, parities punctured with period p; 3 tail steps per encoder. Decoder: max-log-MAP, iterative.
import numpy as np
def rsc(bits):
    x=[];p=[];r0=r1=r2=0
    for u in list(bits)+[None]*3:
        if u is None: u=r0^r1
        a=u^r0^r1; x.append(u); p.append(a^r2^r0); r0,r1,r2=r1,r2,a
    return x,p
def encode(info,perm,p,n):
    k=len(info); x1,p1=rsc(info); x2,p2=rsc([info[i] for i in perm]); out=[]
    for t in range(k):
        out.append(x1[t])
        if p==1 or t%p==0: out.append(p1[t])
        if p==1 or t%p==p//2: out.append(p2[t])
    for t in range(k,k+3): out+=[x1[t],p1[t],x2[t],p2[t]]
    return out+[0]*(n-len(out))
# trellis: state index = r0*4 + r1*2 + r2
NS=np.zeros((8,2),int); PAR=np.zeros((8,2),int)
for s in range(8):
    r0,r1,r2=s>>2,(s>>1)&1,s&1
    for u in (0,1):
        a=u^r0^r1; PAR[s,u]=a^r2^r0; NS[s,u]=(r1<<2)|(r2<<1)|a
TAILU=np.array([(s>>2)^((s>>1)&1) for s in range(8)])
def siso(Ls,Lp,La,Lts,Ltp):
    """Ls, Lp, La: LLRs (positive = bit 0) of systematic, parity and a-priori for k steps; Lts, Ltp: the 3 tail steps. Returns extrinsic + posterior."""
    k=len(Ls); NEG=-1e9
    g=np.empty((k,8,2))
    for u in (0,1): g[:,:,u]=0.5*((1-2*u)*(Ls+La))[:,None]+0.5*(1-2*PAR[:,u])[None,:]*Lp[:,None]
    gt=np.full((3,8,2),NEG)
    for s in range(8):
        u=TAILU[s]; gt[:,s,u]=0.5*(1-2*u)*Lts+0.5*(1-2*PAR[s,u])*Ltp
    G=np.concatenate([g,gt]); T=k+3
    al=np.full((T+1,8),NEG); al[0,0]=0
    for t in range(T):
        c=al[t][:,None]+G[t]; n=np.full(8,NEG); np.maximum.at(n,NS.ravel(),c.ravel()); al[t+1]=n-n.max()
    be=np.full((T+1,8),NEG); be[T,0]=0
    for t in range(T-1,-1,-1):
        c=G[t]+be[t+1][NS]; b=c.max(1); be[t]=b-b.max()
    m=al[:k,:,None]+g+be[1:k+1][:,NS] if False else al[:k,:,None]+g+be[1:k+1][np.arange(k)[:,None,None],NS[None,:,:]]
    post=m[:,:,0].max(1)-m[:,:,1].max(1)
    return post-Ls-La, post
def decode(llr,perm,p,k,iters=8):
    """llr: n soft values in transmission order after de-interleaving (positive = 0). Returns hard info bits."""
    perm=np.asarray(perm); Ls=np.zeros(k); L1=np.zeros(k); L2=np.zeros(k); pos=0
    for t in range(k):
        Ls[t]=llr[pos]; pos+=1
        if p==1 or t%p==0: L1[t]=llr[pos]; pos+=1
        if p==1 or t%p==p//2: L2[t]=llr[pos]; pos+=1
    tail=np.asarray(llr[pos:pos+12]).reshape(3,4); Ls2=Ls[perm]; La=np.zeros(k)
    for _ in range(iters):
        e1,_=siso(Ls,L1,La,tail[:,0],tail[:,1])
        e2,post2=siso(Ls2,L2,e1[perm],tail[:,2],tail[:,3])
        La=np.zeros(k); La[perm]=e2
    out=np.zeros(k); out[perm]=post2
    return (out<0).astype(int)
