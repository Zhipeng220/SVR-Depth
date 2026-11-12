"""CPU-only heartbeat; never used inside timed inference."""
from contextlib import contextmanager
import threading,time

@contextmanager
def cache_io(label,prefix='CACHE IO'):
    stop=threading.Event();start=time.monotonic()
    def tick():
        while not stop.wait(10):print(f'[{prefix}]',label,f'elapsed={time.monotonic()-start:.0f}s; operation still in progress',flush=True)
    thread=threading.Thread(target=tick,daemon=True);thread.start()
    try:yield
    finally:stop.set();thread.join(timeout=1)
