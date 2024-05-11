# -*- coding: utf-8 -*-
from threading import Thread
import queue
from queue import Queue


class ThreadedDataLoader:
    def __init__(self, dataset, max_queue_size, num_threads):
        """
        dataset - явлется генератором, с определенным __len__.
        """
        if num_threads < dataset.batch_size:
            raise ValueError('Количество потоков должно быть больше чем dataset.batch_size')
        
        self.dataset = dataset
        
        self.batches_queue = Queue(maxsize=max_queue_size)
        self.cuda_batches_queue = Queue(maxsize=max_queue_size)
        
        self.threads_feeder = [Thread(target=self.threaded_batches_feeder) for _ in range(num_threads)]
        self.thread_cuda = Thread(target=self.threaded_cuda_batches)
        
        self.thread_killer = ThreadKiller()
        
    def start_threads(self):
        for thread in self.threads_feeder:
            thread.start()
        self.thread_cuda.start()

    def stop_threads(self):
        self.thread_killer.set_tokill(True)
        for thread in self.threads_feeder:
            thread.join()
            print('поток заджоин')
        
        self.thread_cuda.join()
        print('после джоина всех потоков')

    def threaded_batches_feeder(self):
        for batch in self.dataset:
            if self.thread_killer():
                return
            self.batches_queue.put(batch, block=True)
            
            if self.thread_killer():
                return

    def threaded_cuda_batches(self):
        while not self.thread_killer():
            try:
                batch = self.batches_queue.get(block=True)
                self.cuda_batches_queue.put(batch, block=True)
            except queue.Empty:
                pass
            except queue.Full:
                pass
    
    def __len__(self):
        return len(self.dataset) // self.dataset.batch_size
    
    def __iter__(self):
        return self
    
    def __next__(self):
        batch = self.cuda_batches_queue.get(block=True)
        return batch


class ThreadKiller:
    """Boolean object for signaling a worker thread to terminate"""

    def __init__(self):
        self.to_kill = False

    def __call__(self):
        return self.to_kill

    def set_tokill(self, tokill):
        self.to_kill = tokill


def threaded_batches_feeder(tokill, batches_queue, dataset):
    while True:
        for batch in dataset:
            batches_queue.put(batch, block=True)
            if tokill() == True:
                return


def threaded_cuda_batches(tokill, cuda_batches_queue, batches_queue):
    while True:
        batch = batches_queue.get(block=True)
        cuda_batches_queue.put(batch, block=True)

        if tokill() == True:
            return
