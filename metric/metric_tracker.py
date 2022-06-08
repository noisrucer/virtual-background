import pandas as pd

class MetricTracker:
    def __init__(self, mode='train'):
        self.mode = mode + '_' if mode=='val' else ''
        self._log = dict()
        self._reset()

    def update_by_dict(self, log_dict):
        update_dict = {self.mode + k: v for k,v in log_dict.items()}
        self._log.update(update_dict)

    def update_by_key(self, key, value):
        key = self.mode + key
        self._log.update({key, value})

    def result_dict(self):
        return self.log

    def result_df(self):
        return self.dict2df(self.result_dict())

    def ious2dict(self, ious: list):
        print(type(ious))
        cls_names = pd.read_csv('class_dict.csv')['name'].values
        ious_dict = {}

        for cls_name, iou in zip(cls_names, ious):
            ious_dict[cls_name] = iou

        return ious_dict

    def dict2df(self, dct):
        return pd.DataFrame({'value':dct.values()}, index=dct.keys())

    def _reset(self):
        self._log = {}

    @property
    def log(self):
        return self._log
