import os
import pandas as pd
import pprint as pp

from sklearn import metrics
from time import localtime, strftime


class Reporter:
    def __init__(self, algorithm, params, feature_importances, desc):
        self.algorithm = algorithm
        self.params = params
        self.feature_importances = feature_importances
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
            import os
            print(os.path.getcwd())
            df_pred.to_csv('pred.tsv', sep='\t', encoding='utf-8', index=False)

        accuracy = metrics.accuracy_score(y_true, y_pred)

        roc_auc = 0
        if y_prob is not None:
            roc_auc = metrics.roc_auc_score(y_true, y_prob)

        precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_true, y_pred, pos_label=1,
                                                                                     labels=[1, 0])
        cf_matrix = metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])

        r = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'fscore': fscore,
            'support': support,
            'cf_matrix': cf_matrix,
            'desc': desc,
        }
        self.test_results.append(r)
        return

    def write(self):
        now_str = strftime("%Y-%m-%d_%H:%M:%S", localtime())
        output_dir = os.path.join('./report', self.algorithm)
        os.makedirs(output_dir, exist_ok=True)

        output_filename = self.algorithm + '_' + now_str + '_summary.log'

        with open(os.path.join(output_dir, output_filename), 'w') as fo:
            self.print_out(fo)

        return

    def print_out(self, stream=None):
        pp.pprint(self.algorithm, stream=stream)
        pp.pprint('<<Parameters>>', stream=stream)
        pp.pprint(self.test_results, stream=stream)
        pp.pprint('<<Feature Importances>>', stream=stream)
        pp.pprint(self.feature_importances, stream=stream)
        return
