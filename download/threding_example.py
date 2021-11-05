
import threading
import time


def mythread():
    time.sleep(1)


threads = 0     #thread counter
y = 2000     #a MILLION of 'em!
for i in range(y):
    try:
        x = threading.Thread(target=mythread, daemon=True)
        threads += 1    #thread counter
        x.start()       #start each thread
    except RuntimeError:    #too many throws a RuntimeError
        break
print("{} threads created.\n".format(threads))

