import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib import cm


class Plotting:

    def __init__(self, source='/home/mat/Python/data/echo_delineation', ventricle=None):

        self.source = source
        self.data = ventricle.data
        self.id = ventricle.id
        self.number_of_frames = ventricle.number_of_frames
        self.curvature = ventricle.ventricle_curvature
        self.c_normalized = ventricle.vc_normalized
        self.es_frame, self.ed_frame = ventricle.es_frame, ventricle.ed_frame
        self.es_apex = self.data[self.es_frame, ventricle.apex*2:ventricle.apex*2+2]
        self.ed_apex = self.data[self.ed_frame, ventricle.apex*2:ventricle.apex*2+2]
        self.ed_apex_id = ventricle.apex

    def get_translated_element(self, _frame_number, _ref=()):

        if not np.any(_ref):
            x_ref = np.mean(self.data[_frame_number, ::2])
            y_ref = np.mean(self.data[_frame_number, 1::2])
        else:
            x_ref = _ref[0]
            y_ref = _ref[1]
        x_centered = self.data[_frame_number, ::2] - x_ref
        y_centered = self.data[_frame_number, 1::2] - y_ref

        return x_centered, y_centered, (x_ref, y_ref)

    @staticmethod
    def append_missing_curvature_values(curve):
        return np.concatenate([[curve[0]], curve, [curve[-1]]])

    def plot_single_frame(self, frame_number=0):

        xx, yy, _ = self.get_translated_element(frame_number)

        fig, ax0 = plt.subplots(figsize=(5, 8))
        ax0.plot(xx, yy, 'gd-', ms=5)
        ax0.set_title('Case {}, frame number: {}'.format(self.id, frame_number))
        ax0.set_xlim(-40, 45)
        ax0.set_ylim(-45, 55)
        fig.tight_layout()
        fig.savefig(fname=os.path.join(self.source, '{}_frame_{}'.format(self.id, frame_number)))

    def plot_single_frame_with_curvature(self, frame_number=0):

        xx, yy, _ = self.get_translated_element(frame_number)
        curv = self.append_missing_curvature_values(self.c_normalized[frame_number])

        fig, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 5]}, figsize=(13, 8))

        ax0.plot(xx, yy, 'gd-', ms=5)
        ax0.set_title('Case {}, frame number: {}'.format(self.id, frame_number))
        ax0.set_xlim(-40, 45)
        ax0.set_ylim(-45, 55)

        ax1.plot(curv)
        ax1.set_title('Geometric point-to-point curvature')
        ax1.axhline(y=0, color='r', linestyle='--')

        fig.tight_layout()
        fig.savefig(fname=os.path.join(self.source, '{}_frame_{}_with_curv'.format(self.id, frame_number)))

    def plot_all_frames(self, coloring_scheme=None):
        fig, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 5]}, figsize=(13, 8))

        ax0.set_title('Case {}, full cycle'.format(self.id))
        ax0.set_xlim(-30, 55)
        ax0.set_ylim(-5, 95)
        ax0.set_xlabel('Short axis')
        ax0.set_ylabel('Long axis')

        ax1.set_title('Geometric point-to-point curvature')
        ax1.axhline(y=0, c='r', ls='--')
        ax1.set_ylim(-0.08, 0.17)
        ax1.vlines(self.ed_apex_id+1, 0, max(self.curvature[:, self.ed_apex_id]), color='k', linestyles='-.', lw=1)
        #  Added 1 to ed_apex_id because the plot is moved by one (due to lack of curvature at end points)
        ax1.set_xlabel('Point number')
        ax1.set_ylabel('Curvature $[m^{-1}]$')

        if coloring_scheme == 'curvature':
            xx, yy, _ = self.get_translated_element(self.ed_frame, self.ed_apex)
            curv = self.append_missing_curvature_values(self.curvature[self.ed_frame])
            ax0.plot(xx, yy, 'k--', lw=3)
            ax1.plot(curv, '--', c='black', lw=2)

            xx, yy, _ = self.get_translated_element(self.es_frame, self.ed_apex)
            curv = self.append_missing_curvature_values(self.curvature[self.es_frame])
            ax0.plot(xx, yy, 'k:', lw=3)
            ax1.plot(curv, ':', c='black', lw=2)

            legend_elements0 = \
                [Line2D([0], [0], c='k', ls='--', lw=2, label='\'End diastole\''),
                 Line2D([0], [0], c='k', ls=':', lw=2, label='\'End systole\''),
                 Line2D([0], [0], c='w', marker='d', markerfacecolor='k', markersize=9, label='Apex at \'ED\'')]
            legend_elements1 = [Line2D([0], [0], c='k', ls='--', lw=2, label='\'End diastole\''),
                                Line2D([0], [0], c='k', ls=':', lw=2, label='\'End systole\''),
                                Line2D([0], [0], c='b', lw=2, label='Negative curvature'),
                                Line2D([0], [0], c='r', lw=2, label='Positive curvature'),
                                Line2D([0], [0], c='k', ls='-.', label='Apical point')]
        else:
            legend_elements0 = \
                [Line2D([0], [0], c='b', lw=2, label='Beginnning (end diastole)'),
                 Line2D([0], [0], c='purple', lw=2, label='Contraction'),
                 Line2D([0], [0], c='r', lw=2, label='End systole'),
                 Line2D([0], [0], c='g', lw=2, label='Towrds end diastole'),
                 Line2D([0], [0], c='w', marker='d', markerfacecolor='k', markersize=9, label='Apical point')]
            legend_elements1 = [Line2D([0], [0], c='b', lw=2, label='Beginnning'),
                                Line2D([0], [0], c='purple', lw=2, label='Contraction'),
                                Line2D([0], [0], c='r', lw=2, label='End systole'),
                                Line2D([0], [0], c='g', lw=2, label='End'),
                                Line2D([0], [0], c='k', ls=':', label='Apical point')]

        for frame_number in range(self.number_of_frames):

            xx, yy, _ = self.get_translated_element(frame_number, self.ed_apex)
            curv = self.append_missing_curvature_values(self.curvature[frame_number])
            norm_curv = self.append_missing_curvature_values(self.c_normalized[self.ed_frame])
            if coloring_scheme == 'curvature':

                color_tr = cm.coolwarm(norm_curv)
                color_tr[:, -1] = 0.3
                color = cm.seismic(norm_curv)
                size = 10
                ax0.scatter(xx, yy, c=color, edgecolor=color_tr, marker='o', s=size)

                points = np.array([np.linspace(0, len(curv)-1, len(curv)), curv]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                norm = plt.Normalize(-0.125, 0.125)  # Arbitrary values, seem to correspond to the ventricle image
                lc = LineCollection(segments, cmap='seismic', alpha=0.4, norm=norm)
                lc.set_array(curv)
                lc.set_linewidth(2)
                lc.set_edgecolor(color_tr)
                ax1.add_collection(lc)
                ext = 'curvature'
            else:
                color_tr = np.array(cm.brg(frame_number/self.number_of_frames)).reshape((1, -1))[0]
                color_tr[-1] = 0.2
                ax0.plot(xx, yy, c=color_tr, marker='.')
                ax1.plot(curv, c=color_tr, lw=2)
                ext = 'frame'

        ax0.scatter(0, 0, c='k', marker='d', s=80, alpha=1, label='Apex at ED')
        ax0.legend(handles=legend_elements0, loc='upper left', title='Cardiac cycle')
        ax1.legend(handles=legend_elements1, loc='upper right', title='Curvature')
        fig.tight_layout()
        fig.savefig(fname=os.path.join(self.source, '{}_colour_by_{}'.format(self.id, ext)))
