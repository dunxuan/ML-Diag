import os

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, LeavePOut
from skops.io import load


class Classifier:
    def __init__(self, df_file, model_file=None) -> None:
        super().__init__()
        self.df = pd.read_csv(df_file)
        self.model = model_file
        # 删除重复条目
        self.df.drop_duplicates(inplace=True)
        # 填充空值
        # self.df = self.df.groupby('label').apply(lambda x: x.fillna(x.mean()))
        self.df.fillna(self.df.mean(), inplace=True)
        # self.df.fillna(0, inplace=True)
        if not os.path.exists('./Cache/'):
            os.makedirs('./Cache/')

    # 训练
    def train(self, cv_method, cv_params, use_model_params, clf_params):
        X = self.df.drop(['sample_id', 'label'], axis=1)
        y = self.df['label']
        # 数据集分割方法
        if cv_method == '交叉验证法':
            CV = StratifiedKFold(n_splits=cv_params['n_splits'],
                                 shuffle=cv_params['shuffle'],
                                 random_state=cv_params['random_state'])
        elif cv_method == '留一法':
            CV = LeavePOut(p=cv_params['p'])
        # 训练参数
        if self.model is not None:
            clf = load(self.model)
            if not use_model_params:
                clf.set_params(n_estimators=clf_params['n_estimators'],
                               criterion=clf_params['criterion'],
                               max_depth=clf_params['max_depth'],
                               min_samples_split=clf_params['min_samples_split'],
                               min_samples_leaf=clf_params['min_samples_leaf'],
                               min_weight_fraction_leaf=clf_params['min_weight_fraction_leaf'],
                               max_features=clf_params['max_features'],
                               max_leaf_nodes=clf_params['max_leaf_nodes'],
                               min_impurity_decrease=clf_params['min_impurity_decrease'],
                               bootstrap=clf_params['bootstrap'],
                               oob_score=clf_params['oob_score'],
                               n_jobs=clf_params['n_jobs'],
                               random_state=clf_params['random_state'],
                               verbose=clf_params['verbose'],
                               warm_start=clf_params['warm_start'],
                               class_weight=clf_params['class_weight'],
                               ccp_alpha=clf_params['ccp_alpha'],
                               max_samples=clf_params['max_samples'])
        else:
            clf = RandomForestClassifier(n_estimators=clf_params['n_estimators'],
                                         criterion=clf_params['criterion'],
                                         max_depth=clf_params['max_depth'],
                                         min_samples_split=clf_params['min_samples_split'],
                                         min_samples_leaf=clf_params['min_samples_leaf'],
                                         min_weight_fraction_leaf=clf_params['min_weight_fraction_leaf'],
                                         max_features=clf_params['max_features'],
                                         max_leaf_nodes=clf_params['max_leaf_nodes'],
                                         min_impurity_decrease=clf_params['min_impurity_decrease'],
                                         bootstrap=clf_params['bootstrap'],
                                         oob_score=clf_params['oob_score'],
                                         n_jobs=clf_params['n_jobs'],
                                         random_state=clf_params['random_state'],
                                         verbose=clf_params['verbose'],
                                         warm_start=clf_params['warm_start'],
                                         class_weight=clf_params['class_weight'],
                                         ccp_alpha=clf_params['ccp_alpha'],
                                         max_samples=clf_params['max_samples'])
        # 模型训练
        X_CV = X.to_numpy()
        y_CV = y.to_numpy()
        scores = []
        for train, test in CV.split(X_CV, y_CV):
            X_train, X_test, y_train, y_test = X_CV[train], X_CV[test], y_CV[train], y_CV[test]
            # 平衡数据集
            X_train, y_train = SMOTE().fit_resample(X_train, y_train)
            clf.fit(X_train, y_train)
            scores.append(f1_score(y_test, clf.predict(X_test), average='macro'))
        scores = np.array(scores)
        return scores, clf

    # 验证
    def validate(self):
        clf = load(self.model)
        X = self.df.drop(['sample_id', 'label'], axis=1).to_numpy()
        y_true = self.df['label'].to_numpy()
        y_pred = clf.predict(X)
        validate_scores = accuracy_score(y_true, y_pred)
        return y_pred, validate_scores

    # 测试
    def test(self):
        clf = load(self.model)
        X = self.df.drop(['sample_id'], axis=1).to_numpy()
        y_pred = clf.predict(X)
        return y_pred
