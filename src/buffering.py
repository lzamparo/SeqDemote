import multiprocessing as mp
import queue
import threading
import time
 
def buffered_gen_mp(source_gen, buffer_size=2):
    """
    Generator that runs a slow source generator in a separate process.
    buffer_size: the maximal number of items to pre-generate (length of the buffer)
    """
    if buffer_size < 2:
        raise RuntimeError("Minimal buffer size is 2!")
 
    buffer = mp.Queue(maxsize=buffer_size - 1)
    # the effective buffer size is one less, because the generation process
    # will generate one extra element and block until there is room in the buffer.
 
    def _buffered_generation_process(source_gen, buffer):
        for data in source_gen:
            buffer.put(data, block=True)
        buffer.put(None) # sentinel: signal the end of the iterator
        buffer.close() # unfortunately this does not suffice as a signal: if buffer.get()
        # was called and subsequently the buffer is closed, it will block forever.
 
    process = mp.Process(target=_buffered_generation_process, args=(source_gen, buffer))
    process.start()
 
    for data in iter(buffer.get, None):
        yield data


def buffered_gen_threaded(source_gen, buffer_size=2):
    """
    Generator that runs a slow source generator in a separate thread. Beware of the GIL!
    buffer_size: the maximal number of items to pre-generate (length of the buffer)
    """
    if buffer_size < 2:
        raise RuntimeError("Minimal buffer size is 2!")
 
    buffer = queue.Queue(maxsize=buffer_size - 1)
    # the effective buffer size is one less, because the generation process
    # will generate one extra element and block until there is room in the buffer.
 
    def _buffered_generation_thread(source_gen, buffer):
        for data in source_gen:
            buffer.put(data, block=True)
        buffer.put(None) # sentinel: signal the end of the iterator
 
    thread = threading.Thread(target=_buffered_generation_thread, args=(source_gen, buffer))
    thread.daemon = True
    thread.start()

    for data in iter(buffer.get, None):
        yield data

def buffered_gen_woozle(source_gen, buffer_size=2, sleep_time=1):
    """
    Generator that runs a slow source generator in a separate thread.
    buffer_size: the maximal number of items to pre-generate (length of the buffer)
    """
    buffer = queue.Queue(maxsize=buffer_size)

    def _buffered_generation_thread(source_gen, buffer):
        while True:
            # we block here when the buffer is full. There's no point in generating more data
            # when the buffer is full, it only causes extra memory usage and effectively
            # increases the buffer size by one.
            while buffer.full():
                print("DEBUG: buffer is full, waiting to generate more data.")
                time.sleep(sleep_time)

            try:
                data = source_gen.next()
            except StopIteration:
                break

            buffer.put(data)
    
    thread = threading.Thread(target=_buffered_generation_thread, args=(source_gen, buffer))
    thread.setDaemon(True)
    thread.start()
    
    while True:
        yield buffer.get()
        buffer.task_done()