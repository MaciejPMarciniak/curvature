import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
from scipy.stats import kruskal, levene, ttest_ind, normaltest, sem
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import seaborn as sns
from plotting import PlottingDistributions
from LV_edgedetection import check_directory
import statsmodels.api as sm


class StatAnalysis:

    COLUMNS = ['min', 'max', 'avg_min_basal_curv', 'avg_avg_basal_curv', 'min_delta', 'max_delta',
               'amplitude_at_t', 'Series_SOP', 'Patient_ID']

    def __init__(self, input_path, output_path, data_filename):
        self.input_path = input_path
        self.output_path = output_path
        self.data_filename = data_filename
        self.df = self.read_dataframe(index_col='patient_ID')

    def read_dataframe(self, index_col='patient_ID'):
        return pd.read_csv(os.path.join(self.input_path, self.data_filename), index_col=index_col, header=0)

    @staticmethod
    def pop_std(x):
        return x.std(ddof=0)

    def variance_test(self, group_a, group_b):
        print('-----Variance test--------------------------------------------------')
        df_a = self.df[group_a]
        df_b = self.df[group_b]
        t_l, p_l = levene(df_a, df_b)
        print('Statistic: {} and p-value: {} of variance comparison'.format(t_l, p_l))
        print('-----END Variance test----------------------------------------------\n')
        # https://docs.scipy.org/doc/scipy - 0.14.0/reference/generated/scipy.stats.levene.html

    def _check_normality_assumptions(self, feature):

        print('-----Normality test-------------------------------------------------')
        print('This function tests the null hypothesis that a sample comes from a normal distribution. ')
        print('It is based on D’Agostino and Pearson’s [1], [2] test that combines skew and kurtosis '
              'to produce an omnibus test of normality.')
        # https://docs.scipy.org/doc/scipy - 0.14.0/reference/generated/scipy.stats.normaltest.html
        t_nt_control, p_nt_control = normaltest(self.df[feature])
        print('Statistic: {} and p-value: {} of controls normality test'.format(t_nt_control, p_nt_control))
        print('-----END Normality test--------------------------------------------\n')
        return t_nt_control, p_nt_control

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

    def _welchs_t_test(self, covariate, group_a, group_b):

        df_a = self.df[group_a]
        df_b = self.df[group_b]

        print('-----Pairwise t-test--------------------------------------------')
        print('Show that there is a diffence in medians between the groups. Use equal_var = False to perform'
              'the Welch\'s test')
        print('T_test, returning one-sided p-value:')
        t_t, p_val = ttest_ind(df_a, df_b, equal_var=False)
        print('Statistic: {} and p-value: {} of t-test analysis '
              'on control vs htn groups'.format(t_t, p_val))
        print('-----END Pairwise t-test-------------------------------------------\n')

    def perform_analysis(self, covariates=('avg_basal_ED')):
        if self.df is None:
            self.read_dataframe()

        for cov in covariates:
            # Check the basic descriptors
            df_cov = self.df[cov]

            print('----------------------------------------------')
            print('----------------------------------------------')
            print('Univariate tests on covariate {}'.format(cov))
            print('----------------------------------------------')
            # print(self.controls)
            print('Control mean: {} and std: {} of {}'.format(df_cov.mean(),
                                                              df_cov.std(),
                                                              df_cov.describe(),
                                                              cov))
            self._check_normality_assumptions(cov)
            # self._multiple_non_parametric_test()
            # self._welchs_t_test(cov,)

    def plot_histograms(self, covariates=('avg_min_basal_curv',)):
        for cov in covariates:

            sns.distplot(self.df[cov], kde=False, rug=True, color='r')
            plt.legend()
            plt.savefig(os.path.join(self.output_path, '{} histogram.png'.format(cov)))
            plt.close()

        # plt.figure()
        parallel_coordinates(self.df[['label', 'min', 'max', 'avg_min_basal_curv', 'avg_avg_basal_curv',
                                      'min_delta', 'max_delta', 'amplitude_at_t']], 'label')
        plt.savefig(os.path.join(self.output_path, 'covariate_distributions.png'))

    def plot_relations(self, pairs=('min', 'max')):
        plot_tool = PlottingDistributions(self.df, pairs[0][0], output_path=self.output_path)
        for pair in pairs:
            plot_tool.plot_with_labels(pair[0], pair[1])


class StrainAnalysis:

    FACTORS_BASIC = [r'strain_avc_Basal Septal', r'strain_avc_Mid Septal', r'strain_avc_Apical Septal',
                     'max_gls_before_avc', 'max_gls']
    FACTORS_WITH_MW = ['GWI', 'MW_Basal Septal', 'MW_Mid Septal', 'MW_Apical Septal', 'PSS_Basal Septal',
                       'PSS_Mid Septal', 'PSS_Apical Septal', 'strain_avc_Basal Septal', 'strain_avc_Mid Septal',
                       'strain_avc_Apical Septal', 'max_gls_before_avc', 'max_gls']

    def __init__(self, patient_data_path, curvature_results_path, output_path, measurements_filename,
                 twodstrain_filename, patient_data_filename, curvature_filename, merged_data_filename):

        self.patient_data_path = patient_data_path
        self.curvature_results_path = curvature_results_path
        self.output_path = output_path

        self.measurements_filename = measurements_filename
        self.twodstrain_filename = twodstrain_filename
        self.patient_data_filename = patient_data_filename
        self.curvature_filename = curvature_filename

        self.df_meas = pd.read_excel(os.path.join(self.patient_data_path, self.measurements_filename),
                                     index_col='ID', header=0)
        self.df_2ds = pd.read_excel(os.path.join(self.patient_data_path, self.twodstrain_filename),
                                    index_col='ID', header=0)
        if not(os.path.isfile(os.path.join(self.output_path, merged_data_filename))):
            self.get_min_ed_rows(True)
            self.combine_measurements_2ds(True)

        self.df_comparison = pd.read_csv(os.path.join(self.output_path, merged_data_filename),
                                         index_col='patient_ID', header=0)
        self.df_comparison['curv_threshold'] = (self.df_comparison.avg_basal_ED < -0.09).astype(int) #+ 1 # CHANGE SIGN!!!!!!

        self.models = {}

    # ---Processing and combining dataframes----------------------------------------------------------------------------

    def get_min_ed_rows(self, to_file=False):
        """
        Find cases (index level = 0) where the end diastolic trace's curvature is the lowest. Used in case a single
        case has a few views/strain measurements done.
        :param to_file: Whether to save the result to a file.
        """
        df_curv_full = pd.read_csv(os.path.join(self.curvature_results_path, self.curvature_filename),
                                   index_col=['patient_ID', 'patient_ID_detail'], header=0)
        df_curv_full.dropna(inplace=True)

        df_curv = df_curv_full.loc[df_curv_full.groupby(level=0).min_ED.idxmin().values]
        df_curv.reset_index(level=1, inplace=True)

        if to_file:
            df_curv.to_csv(os.path.join(self.output_path, 'curv_min_ED.csv'))

    def combine_measurements_2ds(self, to_file=False):
        """
        Combine information from 3 different sources: 2DStrain (parsed xml export), measurements of WT and curvature
        indices.
        :param to_file: Whether to save the result to a file.
        """

        # exit('Run only if really necessary! If so, update RGM and RFNA (files in export, 1.11.2019)')

        relevant_columns = ['patient_ID_detail', 'min', 'min_delta', 'avg_min_basal_curv', 'avg_avg_basal_curv',
                            'min_ED', 'min_delta_ED', 'avg_basal_ED', 'SB', 'min_index', 'min_index_ED',
                            'strain_avc_Apical Lateral', 'strain_avc_Apical Septal', 'strain_avc_Basal Lateral',
                            'strain_avc_Basal Septal', 'strain_avc_Mid Lateral', 'strain_avc_Mid Septal',
                            'strain_min_Apical Lateral', 'strain_min_Apical Septal', 'strain_min_Basal Lateral',
                            'strain_min_Basal Septal', 'strain_min_Mid Lateral', 'strain_min_Mid Septal',
                            'max_gls_before_avc', 'max_gls', 'max_gls_time', r'IVSd (basal) PLAX', r'IVSd (mid) PLAX',
                            r'PLAX basal/mid', r'IVSd (basal) 4C', r'IVSd (mid) 4C', r'4C basal/mid', 'SB_meas']

        self.df_curv = pd.read_csv(os.path.join(self.output_path, 'curv_min_ED.csv'), header=0, index_col='patient_ID')

        df_meas_2ds = self.df_curv.join(self.df_2ds, how='outer')  # no on= because it's joined on indexes
        df_meas_2ds = df_meas_2ds.join(self.df_meas, how='outer', rsuffix='_meas')
        df_meas_2ds = df_meas_2ds[relevant_columns]

        if to_file:
            df_meas_2ds.index.name = 'patient_ID'
            df_meas_2ds.to_csv(os.path.join(self.output_path, 'Measurements_and_2Dstrain.csv'))

    def plots_wt_and_curvature_vs_markers(self, save_figures=False):

        plot_dir = check_directory(os.path.join(self.output_path, 'plots'))

        x_labels = ['min_ED', 'avg_min_basal_curv', r'PLAX basal/mid', r'4C basal/mid', 'avg_basal_ED']

        for x_label in x_labels:
            for y_label in self.FACTORS_BASIC:

                if x_label in ['PLAX basal_mid', '4C basal_mid']:
                    plt.axvline(1.4, linestyle='--', c='k')
                    self.df_comparison.plot(x=x_label, y=y_label, c='SB', kind='scatter', legend=True, colorbar=True,
                                            cmap='winter', title='Relation of {} to {}'.format(y_label, x_label))
                else:
                    self.df_comparison.plot(x=x_label, y=y_label, c=x_label, kind='scatter', legend=True, colorbar=True,
                                            cmap='autumn', title='Relation of {} to {}'.format(y_label, x_label))
                if save_figures:
                    plt.savefig(os.path.join(plot_dir, r'{} vs {} HTNs.png'.format(y_label, x_label.replace('/', '_'))))
                else:
                    plt.show()
                plt.close()

    def plot_curv_vs_wt(self, save_figures=False, w_reg=False):

        plot_dir = check_directory(os.path.join(self.output_path, 'plots'))
        x_labels = [r'PLAX basal/mid', r'4C basal/mid', 'IVSd (basal) PLAX', 'IVSd (mid) PLAX', 'IVSd (basal) 4C',
                    'IVSd (mid) 4C']
        y_labels = ['min_ED', 'avg_basal_ED', 'avg_min_basal_curv']

        # for x_label in x_labels:
        #     for y_label in y_labels:
        #         self.df_comparison.plot(x=x_label, y=y_label, c='SB', kind='scatter', legend=True, colorbar=True,
        #                                 cmap='winter', title='Relation of {} to {}'.format(y_label, x_label))
        #         means_x = self.df_comparison.groupby('SB')[x_label].mean()
        #         means_y = self.df_comparison.groupby('SB')[y_label].mean()
        #         plt.plot(means_x, means_y, 'kd')
        #
        #         if x_label in [r'PLAX basal/mid', r'4C basal/mid']:
        #             plt.axvline(1.4, linestyle='--', c='k')
        #         if save_figures:
        #             plt.savefig(os.path.join(plot_dir, r'Meas {} vs {} HTNs.png'.format(y_label,
        #                                                                                 x_label.replace('/', '_'))))
        #         else:
        #             plt.show()
        #         plt.close()

        print('Curvature below -1: {}'.format(self.df_comparison.curv_threshold.sum()))
        print('4C above 1.4: {}'.format((self.df_comparison['4C basal/mid'] > 1.4).sum()))
        print('PLAX above 1.4: {}'.format((self.df_comparison['PLAX basal/mid'] > 1.4).sum()))
        print('SB cases: {}'.format((self.df_comparison.SB > 1).sum()))
        self.df_comparison.plot(x=r'PLAX basal/mid', y=r'4C basal/mid', c='curv_threshold', kind='scatter', legend=True,
                                colorbar=True, cmap='winter', title='Curvature value for different wt ratios')
        plt.axvline(1.4, ymax=0.44, linestyle='--', c='k')
        plt.axhline(1.4, xmax=0.43, linestyle='--', c='k')
        if save_figures:
            plt.savefig(os.path.join(plot_dir, r'Ratios_curvature.png'))
        else:
            plt.show()

    def get_statistics(self, indices=()):

        df_stats = pd.DataFrame()

        for marker in self.FACTORS_BASIC:
            df_stats['sb_mean_'+marker] = self.df_comparison.groupby('SB')[marker].mean()
            df_stats['sb_sd_'+marker] = self.df_comparison.groupby('SB')[marker].std()
            df_stats['curv_mean_'+marker] = self.df_comparison.groupby('curv_threshold')[marker].mean()
            df_stats['curv_sd_' + marker] = self.df_comparison.groupby('curv_threshold')[marker].std()
        df_stats.to_csv(os.path.join(self.output_path, 'Simple_statistics.csv'))

    def linear_regression_basic_factors(self, to_file=False, show_plots=False):
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score

        markers = ['avg_avg_basal_curv', 'min_ED', 'avg_basal_ED', r'PLAX basal/mid', r'4C basal/mid']

        list_results = []

        for marker in markers:
            for factor in self.FACTORS_BASIC:

                x = self.df_comparison[marker].values.reshape(-1, 1)
                y = self.df_comparison[factor].values.reshape(-1, 1)

                lr = LinearRegression()
                lr.fit(x, y)
                y_pred = lr.predict(x)

                dict_results = {'marker': marker, 'factor': factor, 'coefficients': lr.coef_, 'R2': r2_score(y, y_pred),
                                'mse': mean_squared_error(y, y_pred)}
                list_results.append(dict_results)

                if show_plots:
                    plots = PlottingDistributions(self.df_comparison, 'min',
                                                  check_directory(os.path.join(self.output_path, 'plots')))
                    plots.plot_with_labels(series1=marker, series2=factor, w_labels=False)

        df_results = pd.DataFrame(list_results)

        if to_file:
            df_results.to_csv(os.path.join(self.output_path, 'Linear_regression_results.csv'))


class VariabilityAnalysis:

    OBSERVERS = ['F1', 'F2', 'M', 'J']
    MEASUREMENTS = ['PLAX basal', 'PLAX mid', '4C basal', '4C mid']

    def __init__(self, measurements_path, output_path, measurements_filename):
        self.measurements_path = measurements_path
        self.output_path = output_path
        self.measurements_filename = measurements_filename

        self.n_samples = 20
        self.df_wt = None
        self.df_curv = None
        self._read_data()

    def _read_data(self):

        df_file = os.path.join(self.measurements_path, self.measurements_filename)
        self.df_wt = pd.read_excel(df_file, sheet_name='WT_measurements', header=[0, 1], index_col=0)
        self.df_curv = pd.read_excel(df_file, sheet_name='Curvature', header=0, index_col='Study_id')
        self.df_test = pd.read_excel(df_file, sheet_name='Sheet2', header=[0, 1])
        self.df_test.columns = pd.MultiIndex.from_tuples(self.df_test.columns)  # Multi header: abs-rel/diff
        self.df_wt.columns = pd.MultiIndex.from_tuples(self.df_wt.columns)  # Multi index header: observer/measurements

    def calculate_sem_multi_index(self, view='PLAX', segment='basal', o1='F1', o2='F2', extended=False):

        assert view in ('PLAX', '4C'), 'Use only PLAX or 4C view'
        assert segment in ('basal', 'mid', 'ratio'), 'Use only basal or mid segment or ratio label'

        colname = ' '.join([view, segment])

        self.df_wt['SEM', 'Mean_measurement'] = (self.df_wt[o1, colname] + self.df_wt[o2, colname]) / 2
        self.df_wt['SEM', 'Difference'] = self.df_wt[o1, colname] - self.df_wt[o2, colname]
        self.df_wt['SEM', 'Difference_round'] = np.round(self.df_wt.SEM.Difference, decimals=2)
        self.df_wt['SEM', 'Difference_prcnt'] = self.df_wt.SEM.Difference / self.df_wt.SEM.Mean_measurement * 100
        self.df_wt['SEM', 'Difference_prcnt_round'] = np.round(self.df_wt.SEM.Difference_prcnt, decimals=2)

        if extended:
            self.df_wt['SEM', 'Abs difference'] = sem((self.df_wt[o1, colname], self.df_wt[o2, colname]), ddof=1) * 2
            self.df_wt['SEM', 'Individual SD'] = sem((self.df_wt[o1, colname], self.df_wt[o2, colname]), ddof=0) * 2

        mean_sem = np.mean(self.df_wt.SEM).round(decimals=2)
        sd_sem = np.std(self.df_wt.SEM).round(decimals=3)
        print(mean_sem, sd_sem)

        return mean_sem, sd_sem

    def calculate_sem_single_index(self, o1='F1', o2='F2', extended=False):

        self.df_curv['Mean_measurement'] = (self.df_curv[o1] + self.df_curv[o2]) / 2
        self.df_curv['Difference'] = self.df_curv[o1] - self.df_curv[o2]
        self.df_curv['Difference_round'] = np.round(self.df_curv.Difference, decimals=2)
        self.df_curv['Difference_prcnt'] = self.df_curv.Difference / self.df_curv.Mean_measurement
        self.df_curv['Difference_prcnt_round'] = np.round(self.df_curv.Difference_prcnt, decimals=2)

        if extended:
            self.df_curv['Abs difference'] = sem((self.df_curv[o1], self.df_curv[o2]), ddof=1) * 2
            self.df_curv['Individual SD'] = sem((self.df_curv[o1], self.df_curv[o2]), ddof=0) * 2

        mean_sem = np.mean(self.df_curv[['Difference', 'Difference_prcnt']])
        sd_sem = np.std(self.df_curv[['Difference', 'Difference_prcnt']])
        print(mean_sem, sd_sem)

        return mean_sem, sd_sem

    def _test_sem_calculations(self):

        self.df_test['SEM', 'Absolute difference'] = sem((self.df_test.Measurement1.m, self.df_test.Measurement2.m), ddof=1) * 2
        self.df_test['SEM', 'Individual SD'] = sem((self.df_test.Measurement1.m, self.df_test.Measurement2.m), ddof=0) * 2
        self.df_test['SEM', 'Difference'] = self.df_test.Measurement1.m - self.df_test.Measurement2.m
        self.df_test['SEM', 'Mean measurement'] = (self.df_test.Measurement1.m + self.df_test.Measurement2.m) / 2
        self.df_test['SEM', 'Difference_round'] = np.round(self.df_test.SEM.Difference, decimals=2)
        print(self.df_test['SEM', 'Difference'])
        print(self.df_test['SEM', 'Mean measurement'])
        self.df_test['SEM', 'Difference_prcnt'] = self.df_test['SEM', 'Difference'] / self.df_test['SEM', 'Mean measurement'] * 100
        self.df_test['SEM', 'Difference_prcnt_round'] = np.round(self.df_test['SEM', 'Difference_prcnt'])
        print(self.df_test[['Absolute intraobserver variability', 'SEM']])
        mean_sem = np.mean(self.df_test.SEM).round(decimals=2)
        sd_sem = np.std(self.df_test.SEM).round(decimals=3)
        print(mean_sem, sd_sem)

    def calculate_standard_error(self, sem_):

        factor = np.sqrt(2 * self.n_samples)  # np.sqrt(2 * n * (m - 1)), m === 2
        se_plus = sem_ * (1 + 1/factor)
        se_minus = sem_ * (1 - 1/factor)
        return se_plus, se_minus, sem_/factor, 1/factor

    def bland_altman_plot_multi_index(self, o1='F1', o2='F2', view='PLAX', segment='basal'):

        assert view in ('PLAX', '4C'), 'Use only PLAX or 4C view'
        assert segment in ('basal', 'mid', 'ratio'), 'Use only basal or mid segment or ratio label'

        cohort1 = self.df_wt[o1, ' '.join([view, segment])]
        cohort2 = self.df_wt[o2, ' '.join([view, segment])]

        if segment == 'ratio':
            self.bland_altman_plot(cohort1, cohort2, title=' '.join(['Observers F1 vs', o2, view, segment]),
                                   measurement='Wall thickness ratio', units='')
        else:
            self.bland_altman_plot(cohort1, cohort2, title=' '.join(['Observers F1 vs', o2, view, segment]),
                                   measurement='Wall thickness', units='[cm]')
        # self.bland_altman_percentage_plot(cohort1, cohort2, title=' '.join(['Wall thickness percent difference: F1 vs', o2, view, segment]))

    def bland_altman_plot_single_index(self, o1='F1', o2='F2'):

        cohort1 = self.df_curv[o1]
        cohort2 = self.df_curv[o2]

        self.bland_altman_plot(cohort1, cohort2, title='Observers F1 vs ' + o2, measurement='Curvature', units='[1/cm]')
        # self.bland_altman_percentage_plot(cohort1, cohort2, title='Curvature % difference F1 vs ' + o2)

    def bland_altman_plot(self, data1, data2, title, measurement, units, *args, **kwargs):
        data1 = np.asarray(data1)
        data2 = np.asarray(data2)
        mean = np.mean([data1, data2], axis=0)
        diff = data1 - data2  # Difference between data1 and data2
        md = np.mean(diff)  # Mean of the difference
        sd = np.std(diff, axis=0)  # Standard deviation of the difference

        plt.scatter(mean, diff, *args, **kwargs)
        plt.axhline(md, color='gray', linestyle='--')
        plt.axhline(md + 1.96 * sd, color='gray', linestyle=':')
        plt.axhline(md - 1.96 * sd, color='gray', linestyle=':')

        _, _, _, se = self.calculate_standard_error(mean)
        ci_loa_height = sd * se
        ci_loa_x = mean.min(), mean.max()

        plt.errorbar(ci_loa_x, [md + 1.96 * sd] * 2,
                     yerr=ci_loa_height, fmt='none',
                     capsize=10, c='r')

        plt.errorbar(ci_loa_x, [md - 1.96 * sd] * 2,
                     yerr=ci_loa_height, fmt='none',
                     capsize=10, c='r')
        plt.title(title)
        plt.xlabel(measurement + ', average value {}'.format(units))
        plt.ylabel(measurement + ', absolute difference {}'.format(units))
        plt.savefig(os.path.join(self.output_path, ' '.join([measurement, view, segment, 'F1_'+o2])))
        plt.close()
        # plt.show()

    def bland_altman_percentage_plot(self, data1, data2, title, *args, **kwargs):
        data1 = np.round(np.asarray(data1), decimals=2)
        data2 = np.round(np.asarray(data2), decimals=2)

        mean = np.mean([data1, data2], axis=0)
        diff = data1 - data2  # Difference between data1 and data2
        print(diff)
        print(mean)
        diff_pr = diff / mean * 100
        md = np.mean(diff_pr)  # Mean of the difference
        sd = np.std(diff_pr, axis=0)  # Standard deviation of the difference

        plt.scatter(mean, diff_pr, *args, **kwargs)
        plt.axhline(md, color='gray', linestyle='--')
        plt.axhline(md + 1.96 * sd, color='gray', linestyle=':')
        plt.axhline(md - 1.96 * sd, color='gray', linestyle=':')

        _, _, _, se = self.calculate_standard_error(mean)
        ci_loa_height = sd * se
        ci_loa_x = mean.min(), mean.max()

        plt.errorbar(ci_loa_x, [md + 1.96 * sd] * 2,
                     yerr=ci_loa_height, fmt='none',
                     capsize=10, c='r')

        plt.errorbar(ci_loa_x, [md - 1.96 * sd] * 2,
                     yerr=ci_loa_height, fmt='none',
                     capsize=10, c='r')
        plt.ylim((-1000, 1000))
        plt.title(title)
        plt.show()


if __name__ == '__main__':

    # VARIABILITY ANALYSIS

    measurements_path = os.path.join('C:/', 'Data', 'ProjectCurvature', 'InterObserverStudy', 'StudyResults')
    output_path = os.path.join('C:/', 'Data', 'ProjectCurvature', 'InterObserverStudy', 'StudyResults')

    measurements_filename = 'InterObserverStudy.xlsx'

    var = VariabilityAnalysis(measurements_path, output_path, measurements_filename)
    var.calculate_sem_single_index()
    var.calculate_sem_multi_index()
    # var._test_sem_calculations()

    for o2 in ['F2', 'M', 'J']:
        view=''
        segment=''
        var.bland_altman_plot_single_index(o2=o2)
    for o2 in ['F2', 'M', 'J']:
        for view in ['PLAX', '4C']:
            for segment in ['basal', 'mid', 'ratio']:
                var.bland_altman_plot_multi_index(o2=o2, view=view, segment=segment)

    # STRAIN ANALYSIS

    # patient_data_path = os.path.join('C:/', 'Data', 'ProjectCurvature', 'PatientData')
    # curvature_results = os.path.join('C:/', 'Data', 'ProjectCurvature', 'Analysis', 'Output')
    # output = check_directory(os.path.join('C:/', 'Data', 'ProjectCurvature', 'Analysis', 'Output', 'Statistics'))
    # measurements = 'AduHeart_Measurements.xlsx'
    # twodstrain = 'AduHeart_Strain_MW.xlsx'
    # curvature = 'master_table_full.csv'
    # patient_info = 'AduHeart_PatientData_Full.xlsx'
    # merged_data = 'Measurements_and_2Dstrain.csv'
    #
    # anal = StrainAnalysis(patient_data_path, curvature_results, output, measurements, twodstrain, patient_info,
    #                       curvature, merged_data)


    # anal.plots_wt_and_curvature_vs_markers(True)
    # anal.plot_curv_vs_wt(True)
    # anal.get_statistics()
    # anal.linear_regression_basic_factors(False, show_plots=True)

    # STATANALYSIS

    # source = os.path.join('c:/', 'Data', 'Pickles', '_Output', 'Final', 'analysis')
    # datafile = 'biomarkers_proper_scale.csv'

    # source = os.path.join('c:/', 'Data', 'Pickles', '_Output', 'Final')
    # datafile = 'all_biomarkers.csv'
    # output = source

    # anal = StatAnalysis(input_path=source, output_path=output, data_filename=datafile)
    # anal.read_dataframe('Patient_ID')
    # Preprocessing
    # anal.describe_quality_assessment()
    # group_anal = anal.get_df_pre_processing(print=True)
    # group_anal.to_csv(os.path.join(output, 'analysis', 'biomarkers_proper_scale.csv'))

    # Analysis
    # anal.plot_histograms(covariates=('min', 'avg_min_basal_curv', 'avg_avg_basal_curv'))
    # anal.plot_relations(pairs=(('min', 'avg_min_basal_curv'), ('min',  'avg_avg_basal_curv'),
    #                            ('avg_min_basal_curv', 'avg_avg_basal_curv')))
    # anal.perform_analysis()
    # anal.predict_with_lr()
