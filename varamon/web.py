# Local web page: open a recording, or listen to an audio input; each transmission is decoded when it ends.
import os, json, threading, tempfile, webbrowser, numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from . import scan as varahf, hffsk
HERE=os.path.dirname(os.path.abspath(__file__)); T=None; lock=threading.Lock()
L=dict(buf=np.zeros(0),base=0,done=0,rate=48000)
def row(q,t):
    fr=q['frame']; c=fr[-3] if len(fr)>=3 else 0
    return dict(t=round(t,2),dur=round(q['dur'],2),kind=q['kind'],level=q['level'],bw=q.get('bw',2300),ok=bool(q['ok']),size=len(fr),score=round(float(q['score']),2),
                reply=c>>6,compressed=(c>>5)&1,last=(c>>1)&1,full=c&1,hex=fr.hex())
def to48(x,rate):
    if rate==48000: return x
    t=np.arange(int(len(x)*48000/rate))*(rate/48000); return np.interp(t,np.arange(len(x)),x)
def feed(samples,rate):
    """live: add audio, decode transmissions that have ended at least 0.8 s ago"""
    L['buf']=np.concatenate([L['buf'],to48(samples,rate)]); x=L['buf']; out=[]
    if len(x)<4*48000: return out
    segs,_=varahf.segments(x)
    for a,b in segs:
        if b>len(x)-38400 or L['base']+a<L['done']: continue
        s=x[max(0,a-96000):b+33600]; r=varahf.decode_segment(s,T)
        if r:
            k,lv,fr,ok,score,bw=r; out.append(row(dict(dur=(b-a)/48000,kind=k,level=lv,ok=ok,frame=fr,score=score,bw=bw),(L['base']+a)/48000))
        L['done']=L['base']+b
    keep=20*48000
    if len(x)>keep: L['base']+=len(x)-keep; L['buf']=x[-keep:]
    return out
class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def send(self,body,kind='application/json',status=200):
        if not isinstance(body,bytes): body=json.dumps(body).encode()
        self.send_response(status); self.send_header('Content-Type',kind); self.send_header('Content-Length',str(len(body))); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if urlparse(self.path).path=='/': self.send(open(os.path.join(HERE,'page.html'),'rb').read(),'text/html; charset=utf-8')
        else: self.send(b'','text/plain',404)
    def do_POST(self):
        u=urlparse(self.path); body=self.rfile.read(int(self.headers.get('Content-Length',0)))
        try:
            if u.path=='/file':
                fd,tmp=tempfile.mkstemp(suffix='.wav'); os.write(fd,body); os.close(fd); x=hffsk.load48(tmp); os.remove(tmp)
                with lock: r=varahf.scan(x,T,report=lambda *a: None)
                self.send(dict(seconds=round(len(x)/48000,1),frames=[row(q,q['t']) for q in r]))
            elif u.path=='/audio':
                rate=int(parse_qs(u.query).get('rate',['48000'])[0]); s=np.frombuffer(body[:len(body)//2*2],'<i2').astype(float)
                with lock: self.send(dict(frames=feed(s,rate),seconds=round((L['base']+len(L['buf']))/48000,1)))
            elif u.path=='/reset':
                with lock: L.update(buf=np.zeros(0),base=0,done=0)
                self.send({})
            else: self.send(b'','text/plain',404)
        except Exception as e: self.send(dict(error=str(e)),status=400)

def serve(port=8792,open_browser=True):
    global T
    print('preparing tables (about a minute the first time) ...'); T=varahf.tables()
    url='http://127.0.0.1:%d/'%port; print('varamon running at %s  (Ctrl+C to stop)'%url)
    if open_browser: threading.Timer(0.5,webbrowser.open,[url]).start()
    try: ThreadingHTTPServer(('127.0.0.1',port),H).serve_forever()
    except KeyboardInterrupt: pass
    return 0
