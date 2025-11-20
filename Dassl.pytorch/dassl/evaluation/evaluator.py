import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix

from .build import EVALUATOR_REGISTRY


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)
        
        #add
        self.cor0 = 0
        self.cor1 = 0
        self.cor2 = 0
        self.cor3 = 0
        self.cor4 = 0
        self.cor5 = 0
        self.cor6 = 0
        self.cor7 = 0
        self.cor8=0
        self.cor9=0
        self.cor10=0

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        #cifar
        # if self._total == 6000:
        #     print('base session:{} / 6000 = {}'.format(self._correct, self._correct/6000))
        #     self.cor0 = self._correct
        # elif self._total == 6500:
        #     self.cor1 = self._correct-self.cor0
        #     print('session1:{} / 500 = {}'.format(self.cor1, self.cor1 / 500))
        # elif self._total == 7000:
        #     self.cor2 = self._correct-self.cor0-self.cor1
        #     print('session2:{} / 500 = {}'.format(self.cor2, self.cor2 / 500))
        # elif self._total == 7500:
        #     self.cor3 = self._correct-self.cor0-self.cor1-self.cor2
        #     print('session3:{} / 500 = {}'.format(self.cor3, self.cor3 / 500))
        # elif self._total == 8000:
        #     self.cor4 = self._correct-self.cor0-self.cor1-self.cor2-self.cor3
        #     print('session4:{} / 500 = {}'.format(self.cor4, self.cor4 / 500))
        # elif self._total == 8500:
        #     self.cor5 = self._correct-self.cor0-self.cor1-self.cor2-self.cor3-self.cor4
        #     print('session5:{} / 500 = {}'.format(self.cor5, self.cor5 / 500))
        # elif self._total == 9000:
        #     self.cor6 = self._correct-self.cor0-self.cor1-self.cor2-self.cor3-self.cor4-self.cor5
        #     print('session6:{} / 500 = {}'.format(self.cor6, self.cor6 / 500))
        # elif self._total == 9500:
        #     self.cor7 = self._correct-self.cor0-self.cor1-self.cor2-self.cor3-self.cor4-self.cor5-self.cor6
        #     print('session7:{} / 500 = {}'.format(self.cor7, self.cor7 / 500))
        # elif self._total == 10000:
        #     self.cor8 = self._correct-self.cor0-self.cor1-self.cor2-self.cor3-self.cor4-self.cor5-self.cor6-self.cor7
        #     print('session8:{} / 500 = {}'.format(self.cor8, self.cor8 / 500))
        # else:
        #     a=0

        if self._total == 2864:
            print('base session:{} / 6000 = {}'.format(self._correct, self._correct/2864))
            self.cor0 = self._correct
        elif self._total == 3143:
            self.cor1 = self._correct-self.cor0
            print('session1:{} / 279 = {}'.format(self.cor1, self.cor1 / 279))
        elif self._total == 3430:
            self.cor2 = self._correct-self.cor0-self.cor1
            print('session2:{} / 287 = {}'.format(self.cor2, self.cor2 / 287))
        elif self._total == 3728:
            self.cor3 = self._correct-self.cor0-self.cor1-self.cor2
            print('session3:{} / 298 = {}'.format(self.cor3, self.cor3 / 298))
        elif self._total == 4028:
            self.cor4 = self._correct-self.cor0-self.cor1-self.cor2-self.cor3
            print('session4:{} / 300 = {}'.format(self.cor4, self.cor4 / 300))
        elif self._total == 4326:
            self.cor5 = self._correct-self.cor0-self.cor1-self.cor2-self.cor3-self.cor4
            print('session5:{} / 298 = {}'.format(self.cor5, self.cor5 / 298))
        elif self._total == 4614:
            self.cor6 = self._correct-self.cor0-self.cor1-self.cor2-self.cor3-self.cor4-self.cor5
            print('session6:{} / 288 = {}'.format(self.cor6, self.cor6 / 288))
        elif self._total == 4911:
            self.cor7 = self._correct-self.cor0-self.cor1-self.cor2-self.cor3-self.cor4-self.cor5-self.cor6
            print('session7:{} / 297 = {}'.format(self.cor7, self.cor7 / 297))
        elif self._total == 5206:
            self.cor8 = self._correct-self.cor0-self.cor1-self.cor2-self.cor3-self.cor4-self.cor5-self.cor6-self.cor7
            print('session8:{} / 295 = {}'.format(self.cor8, self.cor8 / 295))
        elif self._total == 5494:
            self.cor9 = self._correct-self.cor0-self.cor1-self.cor2-self.cor3-self.cor4-self.cor5-self.cor6-self.cor7-self.cor8
            print('session8:{} / 288 = {}'.format(self.cor9, self.cor9 / 288))
        elif self._total == 5794:
            self.cor10 = self._correct-self.cor0-self.cor1-self.cor2-self.cor3-self.cor4-self.cor5-self.cor6-self.cor7-self.cor8-self.cor9
            print('session8:{} / 300 = {}'.format(self.cor10, self.cor10 / 300))
        else:
            a=0


        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.1f}%\n"
            f"* error: {err:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%"
        )

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                print(
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%"
                )
            mean_acc = np.mean(accs)
            print(f"* average: {mean_acc:.1f}%")

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print(f"Confusion matrix is saved to {save_path}")

        return results
