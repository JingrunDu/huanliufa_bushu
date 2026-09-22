"""复用已有 CSV 标量记录，无额外 GPU 取值或损失计算。"""
import csv
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def track_loss_csv(tracker):
    original = csv.writer

    class LossWriter:
        def __init__(self, writer):
            self.writer = writer
            self.columns = None

        def writerow(self, row):
            row = list(row)
            result = self.writer.writerow(row)
            if row and row[0] == 'iteration':
                self.columns = {name: i for i, name in enumerate(row)}
            elif self.columns and 'total_loss' in self.columns:
                tracker.log({'train/iteration': int(row[self.columns['iteration']]),
                             'train/loss': float(row[self.columns['total_loss']])})
            return result

        def writerows(self, rows):
            for row in rows:
                self.writerow(row)

        def __getattr__(self, name):
            return getattr(self.writer, name)

    def writer(file, *args, **kwargs):
        wrapped = original(file, *args, **kwargs)
        if Path(str(getattr(file, 'name', ''))).name == 'loss_log.csv':
            return LossWriter(wrapped)
        return wrapped

    csv.writer = writer
    try:
        yield
    finally:
        csv.writer = original
