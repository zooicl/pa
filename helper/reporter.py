import pandas as pd
from sklearn import metrics


class Reporter:
    def __init__(self, algorithm, params, desc):
        self.algorithm = algorithm
        self.params = params
        self.desc = desc
        self.test_results = []

        return

    def add(self, y_true, y_pred, y_prob, desc, y_id=None):
        if y_id is not None:
            pred = [
                ('y_id', y_id),
                ('y_true', y_true),
                ('y_pred', y_pred),
                ('y_prob', y_prob),
            ]
            df_pred = pd.DataFrame.from_items(pred)
            df_pred.to_csv('pred.tsv', sep='\t', encoding='utf-8', index=False)

        accuracy = metrics.accuracy_score(y_true, y_pred)
        precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_true, y_pred, pos_label=1,
                                                                                     labels=[1, 0])
        roc_auc = metrics.roc_auc_score(y_true, y_prob)
        cf_matrix = metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])

        r = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'fscore': fscore,
            'support': support,
            'roc_auc': roc_auc,
            'cf_matrix': cf_matrix,
            'desc': desc,
        }
        print(r)

        return

    def write(self):
        return

    def print(self):
        return
