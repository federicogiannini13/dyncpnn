import math
import numpy as np


class SentinelSimulator:
    """
    It simulates a detector with a specific precision and recall
    """

    def __init__(self, precision, recall, dataset, max_delay=128 * 50):
        """

        Parameters
        ----------
        precision: float
            The desired precision of the detector.
        recall: float
            The desired recall of the detector.
        dataset: str
            The name of the dataset It must contain a substring in {'air_quality', 'energy', 'sine', 'weather'}.
        max_delay: int, default: 128*50
            The minimum number of data points following a true drift after which the detection is considered a false
            positive.
        """
        self.cont = 0
        self.precision = precision
        self.recall = recall
        dataset = dataset.lower()

        self.drifts_real = []
        self.dataset_len = 0
        if "weather" in dataset:
            self.drifts_real = [22095, 44200, 66305, 88412, 110520, 132628, 154736]
            self.dataset_len = 176844
        elif "air_quality" in dataset:
            self.drifts_real = [32654, 65318, 97981, 130644, 163307, 195970, 228633]
            self.dataset_len = 261296
        elif "energy" in dataset:
            self.drifts_real = [25646, 51312, 76978, 102644]
            self.dataset_len = 128311
        elif "sine" in dataset:
            self.drifts_real = [30000, 60000, 90000, 120000, 150000, 180000, 210000]
            self.dataset_len = 240000
        self.drifts_real_ranges = [(d, d + max_delay) for d in self.drifts_real]

        self.drifts_real_wrong_ranges = (
            [(0, self.drifts_real[0])]
            + [
                (self.drifts_real[i] + max_delay, self.drifts_real[i + 1])
                for i in range(len(self.drifts_real) - 1)
            ]
            + [(self.drifts_real[-1] + max_delay, self.dataset_len)]
        )
        self.true_positives = math.ceil(self.recall * len(self.drifts_real))
        self.detections = math.floor(self.true_positives / self.precision)
        self.false_positives = self.detections - self.true_positives
        self.drifts = []
        for _ in range(self.true_positives):
            i = np.random.choice(np.arange(0, len(self.drifts_real_ranges)))
            d = self.drifts_real_ranges[i]
            self.drifts.append(np.random.randint(d[0], d[1]))
        for _ in range(self.false_positives):
            i = np.random.choice(np.arange(0, len(self.drifts_real_wrong_ranges)))
            d = self.drifts_real_wrong_ranges[i]
            self.drifts.append(np.random.randint(d[0], d[1]))
        self.drift_detected = False

    def update(self, *args, **kwargs):
        """
        Call this method after each data point to check for detections. It returns True in the case of a detection,
        False otherwise.
        """
        self.drift_detected = False
        if self.cont in self.drifts:
            self.drift_detected = True
        self.cont += 1
        return self.drift_detected
