import pandas as pd
import numpy as np
from scipy.stats import spearmanr, levene, shapiro, mannwhitneyu, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import os


def read_data(meas, param):
    df_meas = pd.read_csv(meas, index_col='patient_ID')
    df_param = pd.read_csv(param, index_col='AAAID_paciente')

    df_clic = df_meas.join(df_param, how='inner')
    df_clic['conduit_contractile_ratio'] = df_clic['LA_contractile_strain'] / df_clic['LA_conduit_strain']
    return df_clic


def _save_plot(plot_name, plot_data, output_path):
    plot_data.figure.savefig(os.path.join(output_path, plot_name))
    plt.close()


def plot_distribution(df, variable, show=False):

    plot_name = 'distribution_' + variable
    _ = plt.figure(figsize=(8, 5))
    _ = sns.distplot(df[variable], label=str(variable), kde=False, hist=True, rug=True, bins=16)
    _ = plt.gcf()

    if show:
        plt.show()
    _save_plot(plot_name + '.svg', _, r'C:\Data\ProjectCurvature\Analysis\Output_HTN\Statistics\AdditionalAnalysis')


def plot_multiple_distributions(df, variable, group_by, show=False):
    unique_labels = np.sort(df[group_by].unique())
    plot_name = 'distribution_' + variable + '_by_' + group_by
    _ = plt.figure(figsize=(8, 5))
    for label in unique_labels:
        group = df[df[group_by] == label]
        _ = sns.distplot(group[variable], label=str(label), hist=True, kde=False, rug=True)

    _ = plt.gcf()

    if show:
        plt.show()
    _save_plot(plot_name + '_20_perc.svg', _,
               r'C:\Data\ProjectCurvature\Analysis\Output_HTN\Statistics\AdditionalAnalysis')


def plot_histograms(df, variables, group_by, show=False):
    for variable in variables:
        _ = plt.figure(figsize=(8, 5))

        _ = sns.distplot(df.loc[df[group_by] == 1][variable], kde=False, rug=True, color='orange')
        _ = sns.distplot(df.loc[df[group_by] == 2][variable], kde=False, rug=True, color='red')
        # plt.legend()

        if 'Average septal curvature' in variable:
            plt.xlabel(r'Average septal curvature $[dm^{-1}]$', fontsize=23)
            plt.ylabel('Frequency', fontsize=30)
        else:
            plt.ylabel('')
            plt.xlabel(variable, fontsize=23)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.tight_layout()
        if show:
            plt.show()

        plot_name = 'distribution_' + variable + '_by_' + group_by
        _save_plot(plot_name + '_20_perc.svg', _,
                   r'C:\Data\ProjectCurvature\Analysis\Output_HTN\Statistics\AdditionalAnalysis')
        plt.clf()
        plt.close()


def check_homoscedastisity(df, variables, grouping):

    for g in grouping:
        for v in variables:
            print(g, v, levene(df.loc[df[g] == 1][v], df.loc[df[g] == 2][v]))


def check_normality(df, variables, grouping):

    for g in grouping:
        for v in variables:
            print(g+'==1', v, shapiro(df.loc[df[g] == 1][v]))
            print(g+'==2', v, shapiro(df.loc[df[g] == 2][v]))


def hyp_test(df, variables, grouping):

    for g in grouping:
        for v in variables:
            if 'Basal' in v or 'Mid' in v or 'e_prime_average' in v:
                print(g, v, mannwhitneyu(df.loc[df[g] == 1][v], df.loc[df[g] == 2][v]))
            else:
                print(g, v, mannwhitneyu(df.loc[df[g] == 1][v], df.loc[df[g] == 2][v]))


measurements = r'C:\Data\ProjectCurvature\Analysis\Output_HTN\Statistics\Measurements_and_2DstrainAdd.csv'
parameters = r'C:\Data\ProjectCurvature\Analysis\Output_HTN\Statistics\BSH_selected_clinical_dataAdd.csv'

vars = ['Average septal curvature [cm-1]', 'Wall thickness ratio in PLAX view', 'Wall thickness ratio in 4CH view',
        'SB_curv', 'SB']
tars = ['strain_avc_Basal Septal', 'strain_avc_Mid Septal', 'LA_contractile_strain', 'LA_conduit_strain', 'E_A_ratio',
        'e_prime_medial', 'a_prime_medial', 'conduit_contractile_ratio',
        'e_prime_lateral','LV_mass_indexed', 'VOL_MAX_INDEX', 'VOL_MIN_INDEX', '2D_LA_conduit_volume'
        ]  # '2D_LA_conduit_volume' -> not relevant in both cases

df_comb = pd.read_csv(measurements)

# Plot distributions
# plot_distribution(df_comb, 'Average septal curvature [cm-1]', show=True)
# plot_histograms(df_comb, tars, 'SB', show=False)
# plot_histograms(df_comb, tars, 'SB_curv', show=False)

# Run statistics
# check_homoscedastisity(df_comb, tars, ['SB', 'SB_curv'])  # -> RESULT: all except e_prime_medial in both cases
# check_normality(df_comb, tars, ['SB', 'SB_curv'])  # -> RESULT: SB_curv: none except e_prime_avg and strains (group 1),
#                                                    # Same in SB, but here also group 2 is sometimes not normal.
# hyp_test(df_comb, tars, ['SB', 'SB_curv'])

# Create grouping for curvature
# print(df_comb['Average septal curvature [cm-1]'].quantile(0.20))
# print(df_comb['Average septal curvature [cm-1]'].quantile(0.85))
# df_comb['SB_curv'] = df_comb['Average septal curvature [cm-1]'].map(lambda x: 2 if
# x<=df_comb['Average septal curvature [cm-1]'].quantile(0.20) else 1)
# print(df_comb['SB_curv'])
# df_comb.to_csv(r'C:\Data\ProjectCurvature\Analysis\Output_HTN\Statistics\Measurements_and_2DstrainAdd.csv')


# Correlation
res = pd.DataFrame(columns=vars, index=tars)
for t in tars:
    for v in vars:
        res.loc[t, v], res.loc[t, v + ' pval'] = spearmanr(df_comb[v], df_comb[t], nan_policy='omit')
#
#
res.to_csv(r'C:\Data\ProjectCurvature\Analysis\Output_HTN\Statistics\results_w_negative.csv')