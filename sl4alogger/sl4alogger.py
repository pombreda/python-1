#import android
import time
import threading
import logging

logging.basicConfig(level=logging.DEBUG, \
        format="[%(levelname)s] (%(threadName)-10s) %(message)s",\
        )
HZ = 10
dt = 1./float(HZ)
droid = android.Android()


class imu_thread(threading.Thread):
    def __init__(self, name):
        droid.startSensingTimed(1, int(dt*1000))
    
    def run(self):
        droid.sensorsReadAccelerometer().result
        time.sleep(dt)
    
    def __del__(self):
        droid.stopSensing()

class camera_thread(threading.Thread):
    loc = '/sdcard/expt/'
    def __init__(self, name):
        droid.recorderCaptureVideo(loc, 100, recordAudio=False)

    def run(self):
        pass

if __name__=='__main__':
    threads = []
    for f in [imu_thread, camera_thread]:
        t = threading.Thread(name=f.__name__, target=f)
        threads.append(t)
        t.start()
