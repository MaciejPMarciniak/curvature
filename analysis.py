import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
from scipy.stats import kruskal, levene, ttest_ind, normaltest
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import seaborn as sns
from plotting import PlottingDistributions


class StatAnalysis:

    COLUMNS = ['min', 'max', 'avg_min_basal_curv', 'avg_avg_basal_curv', 'min_delta', 'max_delta',
               'amplitude_at_t', 'Series_SOP', 'Patient_ID']

    def __init__(self, input_path, output_path, data_filename):
        self.input_path = input_path
        self.output_path = output_path
        self.data_filename = data_filename

        self.df = None
        self.controls = None
        self.htn = None
        self.bsh = None

    def read_dataframe(self, index_col='cycle_id'):
        self.df = pd.read_csv(os.path.join(self.input_path, self.data_filename), index_col=index_col,
                              header=0)
        print(self.df['avg_min_basal_curv'])
        self.df = self.df.drop('TTA0246')
        print(self.df.size)

    def _inclusion_counts(self):
        n_all = len(self.df.index)
        n_faulty_segmentations = len(self.df.loc[self.df['min'] == 0.0, :].index)
        n_faulty_contours = len(self.df.loc[self.df['min'] == 1.0, :].index)
        return n_all, n_faulty_segmentations, n_faulty_contours

    def _inclusion_statistics(self):
        n_all, n_faulty_segmentations, n_faulty_contours = self._inclusion_counts()
        pr_faulty_segmentations = n_faulty_segmentations / n_all * 100
        pr_faulty_contours = n_faulty_contours / n_all * 100
        pr_faulty_all = pr_faulty_contours + pr_faulty_segmentations
        return pr_faulty_all, pr_faulty_segmentations, pr_faulty_contours

    def describe_quality_assessment(self):
        faulty_percentages = self._inclusion_statistics()
        print('Number of cycles: {}\n'
              'Discarded cycles (%): {}\n'
              'Cycles discarded due to faulty segmentations (%): {}\n'
              'Cycles discarded due to faulty segmentations '
              'and contouring (%): {}\n'.format(len(self.df.index), *faulty_percentages))

    def get_df_pre_processing(self):
        df_pre = self.df[self.df['min'] != 0]
        df_pre = df_pre[df_pre['min'] != 1]
        return df_pre.groupby(by=['Patient_ID', 'Series_SOP']).mean()

    def _check_normality_assumptions(self):
        # https://docs.scipy.org/doc/scipy - 0.14.0/reference/generated/scipy.stats.levene.html

        print('-----Variance test--------------------------------------------------')
        t_l, p_l = levene(self.controls, self.htn, self.bsh)
        print('Statistic: {} and p-value: {} of variance comparison'.format(t_l, p_l))
        print('-----END Variance test----------------------------------------------\n')

        print('-----Normality test-------------------------------------------------')
        print('This function tests the null hypothesis that a sample comes from a normal distribution. ')
        print('It is based on D’Agostino and Pearson’s [1], [2] test that combines skew and kurtosis '
              'to produce an omnibus test of normality.')
        # https://docs.scipy.org/doc/scipy - 0.14.0/reference/generated/scipy.stats.normaltest.html
        t_nt_control, p_nt_control = normaltest(self.controls)
        print('Statistic: {} and p-value: {} of controls normality test'.format(t_nt_control, p_nt_control))
        t_nt_htn, p_nt_htn = normaltest(self.htn)
        print('Statistic: {} and p-value: {} of htn normality test'.format(t_nt_htn, p_nt_htn))
        t_nt_bsh, p_nt_bsh = normaltest(self.bsh)
        print('Statistic: {} and p-value: {} of bsh normality test'.format(t_nt_bsh, p_nt_bsh))
        print('-----END Normality test--------------------------------------------\n')
        return p_l, p_nt_control, p_nt_htn, p_nt_bsh

    def _multiple_non_parametric_test(self):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html

        t_k, p_k = kruskal(self.controls, self.htn, self.bsh)
        print('-----Kruskal-Willis--------------------------------------------------')
        print('Comparison across all groups')
        print('Shows that there is a significance in differences among the distribution.')
        print('Null hypothesis: distribution(controls) == distribution(htn) == distribution(bsh)')
        print('Statistic: {} and p-value: {} of kruskal analysis'.format(t_k, p_k))
        print('-----END Kruskal-Willis----------------------------------------------\n')
        return p_k

    def _log_transform_data(self, covariate='', separate_sets=True):

        _df = self.df
        _df[covariate] = np.abs(_df[covariate])
        _df[covariate] = np.log(_df[covariate])
        if separate_sets:
            _controls = _df[_df['label'] == 0][covariate]
            _htn = _df[_df['label'] == 1][covariate]
            _bsh = _df[_df['label'] == 2][covariate]
            return _controls, _htn, _bsh
        else:
            return _df

    def _welchs_t_test(self, covariate):

        controls, htn, bsh = self._log_transform_data(covariate=covariate)

        print('-----Pairwise t-test--------------------------------------------')
        print('Show that there is a diffence in medians between the groups. Use equal_var = False to perform'
              'the Welch\'s test')
        print('T_test, returning one-sided p-value:')
        t_t_control_htn, p_control_htn = ttest_ind(htn, controls, equal_var=False)
        print('Statistic: {} and p-value: {} of t-test analysis '
              'on control vs htn groups'.format(t_t_control_htn, p_control_htn))
        t_t_control_bsh, p_control_bsh = ttest_ind(bsh, controls, equal_var=False)
        print('Statistic: {} and p-value: {} of t-test analysis '
              'on control vs bsh groups'.format(t_t_control_bsh, p_control_bsh))
        t_t_htn_bsh, p_htn_bsh = ttest_ind(bsh, htn, equal_var=False)
        print('Statistic: {} and p-value: {} of t-test analysis '
              'on htn vs bsh groups'.format(t_t_htn_bsh, p_htn_bsh, fmt='%.5'))
        print('-----END Pairwise t-test-------------------------------------------\n')

    def perform_analysis(self, covariates=('min', 'avg_min_basal_curv', 'avg_avg_basal_curv')):
        if self.df is None:
            self.read_dataframe()

        for cov in covariates:
            # Check the basic descriptors
            self.controls = self.df[self.df['label'] == 0][cov]
            self.htn = self.df.loc[self.df['label'] == 1][cov]
            self.bsh = self.df.loc[self.df['label'] == 2][cov]

            print('----------------------------------------------')
            print('----------------------------------------------')
            print('Univariate tests on covariate {}'.format(cov))
            print('----------------------------------------------')
            # print(self.controls)
            print('Control mean: {} and std: {} of {}'.format(self.controls.mean(),
                                                              self.controls.std(),
                                                              cov))
            print('Htn mean: {} and std: {} of {}'.format(self.htn.mean(),
                                                          self.htn.std(),
                                                          cov))
            print('Bsh mean: {} and std: {} of {}'.format(self.bsh.mean(),
                                                          self.bsh.std(),
                                                          cov))
            self._check_normality_assumptions()
            self._multiple_non_parametric_test()
            self._welchs_t_test(covariate=cov)

    def predict_with_lr(self):

        # Log transform data:
        _df = self.df
        _df = self._log_transform_data('avg_min_basal_curv', False)
        _df = self._log_transform_data('avg_avg_basal_curv', False)
        _df = self._log_transform_data('min', False)

        print(_df[['avg_min_basal_curv', 'avg_avg_basal_curv', 'min']])
        # _df = _df[_df['label'] > 0]
        # print(log_df)
        print('Split data')
        X = np.array(_df[['avg_min_basal_curv', 'avg_avg_basal_curv', 'min']].values)  # .reshape(-1, 1)
        y = np.array(_df['label'].values)
        # y[y == 1] = 0
        # y[y == 2] = 1
        # X = X.reshape(-1,1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # X_train.reshape(-1, 1)
        # X_test.reshape(-1, 1)
        print('Define a pipeline to search for the best classifier regularization')
        logistic = SGDClassifier(loss='log', penalty='l2', early_stopping=True,
                                 max_iter=10000, tol=1e-5)
        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        param_grid = {'alpha': np.logspace(-3, 3, 13)}
        search = GridSearchCV(logistic, param_grid, iid=False, cv=5,
                              return_train_score=False, n_jobs=-1, error_score='raise')
        search.fit(X_train, y_train)

        print("Best parameter {}, CV score = {}".format(search.best_params_, search.best_score_*100))
        print('Accuracy score with logistic regression')
        y_pred = search.predict(X_test)
        print('Test set: {}'.format(accuracy_score(y_test, y_pred)))
        # Turn one class to 1 and rest to zero, then apply roc curve
        # print(roc_auc_score(y_test, y_pred, average='samples'))

    def plot_histograms(self, covariates=('min',)):
        for cov in covariates:

            plot_tool = PlottingDistributions(self.df, cov, output_path=self.output_path)
            plot_tool.plot_multiple_distributions(group_by='label')
            plot_tool.plot_multiple_boxplots(group_by='label')
            sns.distplot(self.df.loc[self.df['label'] == 0, cov], kde=False, label='control', rug=True, color='r', bins=12)
            sns.distplot(self.df.loc[self.df['label'] == 1, cov], kde=False, label='htn', rug=True, color='g', bins=15)
            sns.distplot(self.df.loc[self.df['label'] == 2, cov], kde=False, label='bsh', rug=True, color='b', bins=10)
            plt.legend()
            plt.savefig(os.path.join(self.output_path, '{} histogram.png'.format(cov)))
            plt.clf()

        # plt.figure()
        parallel_coordinates(self.df[['label', 'min', 'max', 'avg_min_basal_curv', 'avg_avg_basal_curv',
                                      'min_delta', 'max_delta', 'amplitude_at_t']], 'label')
        plt.savefig(os.path.join(self.output_path, 'covariate_distributions.png'))

    def plot_relations(self, pairs=(('min', 'max'))):
        plot_tool = PlottingDistributions(self.df, pairs[0][0], output_path=self.output_path)
        for pair in pairs:
            plot_tool.plot_with_labels(pair[0], pair[1])


if __name__ == '__main__':
    source = os.path.join('c:/', 'Data', 'Pickles', '_Output', 'Final', 'analysis')
    # source = os.path.join('c:/', 'Data', 'Pickles', '_Output', 'Final')
    output = source
    datafile = 'biomarkers_proper_scale.csv'
    anal = StatAnalysis(input_path=source, output_path=output, data_filename=datafile)
    anal.read_dataframe('Patient_ID')
    # Preprocessing
    # anal.describe_quality_assessment()
    # group_anal = anal.get_df_pre_processing()
    # group_anal.to_csv(os.path.join(output, 'analysis', 'biomarkers_proper_scale.csv'))

    # Analysis

    # anal.plot_histograms(covariates=('min', 'avg_min_basal_curv', 'avg_avg_basal_curv'))
    # anal.plot_relations(pairs=(('min', 'avg_min_basal_curv'), ('min',  'avg_avg_basal_curv'),
    #                            ('avg_min_basal_curv', 'avg_avg_basal_curv')))
    anal.perform_analysis()
    # anal.predict_with_lr()
