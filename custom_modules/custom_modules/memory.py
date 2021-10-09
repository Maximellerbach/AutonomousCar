import threading
import time

import numpy as np

from .datasets.dataset_json import Dataset


class Memory:
    def __init__(self, dataset: Dataset, dos_save, queue_size=50, sleep_time=0.001):
        """"Memory class to save images and annotations using a thread and a queue"

        Args:
            dataset (Dataset): Dataset class
            dos_save ([type]): dos to save the data to
            queue_size (int, optional): queue size. Defaults to 50.
            sleep_time (float, optional): sleep time for the thread pool. Defaults to 0.001.
        """
        self.memory = []  # stored batch of memory
        self.dataset = dataset
        self.dos_save = dos_save

        self.queue_size = queue_size
        self.sleep_time = sleep_time

        self.thread = threading.Thread(target=self.loop)

        self.running = True
        self.saving = False

    def add(self, img: np.array, annotation_dict: dict):
        """Add a given image and annotation to the queue.

        Args:
            img (np.array): The image
            annotation_dict (dict): The annotations for the image
        """
        if len(self.memory) < self.queue_size:
            self.memory.append((img, annotation_dict))
        else:
            self.remove(0)
            self.memory.append((img, annotation_dict))
        return

    def remove(self, index=0):
        """Remove the tuple of image and annotation in the queue at a given index

        Args:
            index (int, optional): index. Defaults to 0.

        Raises:
            IndexError: [description]
        """
        if len(self.memory) < index:
            raise IndexError
        else:
            del self.memory[index]

    def loop(self):
        while(self.running):
            if len(self.memory):
                self.saving = True
                img_to_save, annotation_dict = self.memory[-1]

                self.dataset.save_img_and_annotation(img_to_save, annotation_dict, dos=self.dos_save)
                self.remove(-1)
            else:
                self.saving = False
                time.sleep(self.sleep_time)

    def run(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False


if __name__ == "__main__":
    # just some tests
    dataset = Dataset(["direction", "speed", "throttle", "time"])
    mem = Memory(10)

    mem.run()
