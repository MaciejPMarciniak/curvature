import numpy as np
import pandas as pd
from scipy.stats import kruskal, levene, ttest_ind, shapiro
import os


class StatAnalysis:

    COLUMNS = ['min', 'max', 'min', 'max', 'min_delta', 'max_delta', 'amplitude_at_t', 'Series_SOPID',
               'Patient_ID']

    def __init__(self, input_path, output_path, data_filename):
        self.input_path = input_path
        self.output_path = output_path
        self.data_filename = data_filename

        self.df = None

    def read_dataframe(self):
        self.df = pd.read_csv(os.path.join(self.input_path, self.data_filename), index_col='cycle_id',
                              header=0)

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
              'Cycles discarded due to faulty contouring (%): {}\n'.format(len(self.df.index),
                                                                           *faulty_percentages))

    def get_df_pre_processing(self):
        df_pre = self.df[self.df['min'] != 0]
        df_pre = df_pre[df_pre['min'] != 1]
        return df_pre.groupby(by='Patient_ID').mean()

    def perform_analysis(self):
        df = self.get_df_pre_processing()
        print(df)

        exit()
        # Check the basic descriptors
        controls = self.df.loc['label' == 0, 'min']
        htn = self.df.loc['label' == 1, 'min']
        bsh = self.df.loc['label' == 2, 'min']

        # Variance comparison:
        # https: // docs.scipy.org / doc / scipy - 0.14.0 / reference / generated / scipy.stats.levene.html
        t_l, p_l = levene(controls, htn, bsh)
        print('Statistic: {} and p-value: {} of variance comparison'.format(t_l, p_l))

        #  Check for normality
        from statsmodels.formula.api import ols
        results = ols('label ~ C(min)', data=self.df).fit()
        results.summary()  # Check which label is taken into account

        t_sw, p_sw = shapiro(results.resid)
        print('Statistic: {} and p-value: {} of normality check'.format(t_sw, p_sw))
        # https: // www.statsmodels.org / stable / index.html

        # Get the statistics on differences between distributions
        # Ordinal response variable:

        t_k, p_k = kruskal(controls, htn, bsh)  # comparison across all groups.
        # Shows that there is a significance in differences among the distributions.

        print('Null hypothesis: distribution(controls) == distribution(htn) == distribution(bsh)')
        print('Statistic: {} and p-value: {} of kruskal analysis'.format(t_k, p_k))
        # https: // docs.scipy.org / doc / scipy / reference / generated / scipy.stats.kruskal.html

        # Show that there is a diffence in medians between the groups. Use equal_var = False.
        # T_test, returning one-sided p-value:

        # make sure that # mean_x < mean_y (more diseased come first!) - NOT--------------------------
        t_t_control_htn, p_control_htn = ttest_ind(controls, htn, equal_var=False)
        print('Statistic: {} and p-value: {} of t-test analysis '
              'on control vs htn groups'.format(t_t_control_htn, p_control_htn))
        t_t_control_bsh, p_control_bsh = ttest_ind(controls, bsh, equal_var=False)
        print('Statistic: {} and p-value: {} of t-test analysis '
              'on control vs bsh groups'.format(t_t_control_bsh, p_control_bsh))
        t_t_htn_bsh, p_htn_bsh = ttest_ind(htn, bsh, equal_var=False)
        print('Statistic: {} and p-value: {} of t-test analysis '
              'on htn vs bsh groups'.format(t_t_htn_bsh, p_htn_bsh))

        # p2 / 2 < alpha  # one-sided Welsh t-test, (alpha == 0.05 or 0.005 hopefully)
        # t < 0
        # https: // docs.scipy.org / doc / scipy / reference / generated / scipy.stats.ttest_ind.html

        # Finally, the test:

    def plot_histograms(self):
        bin_values = np.arange(start=-50, stop=200, step=10)
        # us_mq_airlines_index = data['unique_carrier'].isin(
        #     ['US', 'MQ'])  # create index of flights from those airlines
        # us_mq_airlines = data[us_mq_airlines_index]  # select rows
        # group_carriers = us_mq_airlines.groupby('unique_carrier')[
        #     'arr_delay']  # group values by carrier, select minutes delayed
        # group_carriers.plot(kind='hist', bins=bin_values, figsize=[12, 6], alpha=.4,
        #                     legend=True)  # alpha for transparency


if __name__ == '__main__':
    source = os.path.join('c:/', 'Data', 'Pickles', '_Output')
    output = os.path.join(source, 'analysis')
    datafile = 'all_biomarkers.csv'
    anal = StatAnalysis(input_path=source, output_path=output, data_filename=datafile)
    anal.read_dataframe()
    anal.describe_quality_assessment()
    anal.perform_analysis()

