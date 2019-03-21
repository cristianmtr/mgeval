#!/usr/bin/env python
# coding: utf-8
import output
import os
import midi
import glob
import numpy as np
import random
import pretty_midi
import seaborn as sns
import matplotlib.pyplot as plt
from mgeval import core, utils
from sklearn.model_selection import LeaveOneOut
# plt.style.use('ggplot')

# ```
# 'avg_IOI',
# 'avg_pitch_shift',
# 'bar_pitch_class_histogram',
# 'bar_used_note',
# 'bar_used_pitch',
# 'chord_dependency',
# 'note_length_hist',
# 'note_length_transition_matrix',
# 'pitch_class_transition_matrix',
# 'pitch_range',
# 'total_pitch_class_histogram',
# 'total_used_note',
# 'total_used_pitch'
# ```

# ## Absolute measurement: statistic analysis
#


def main(set1, set2, set1name, set2name, dstfolder):

    set1_eval = {
        'total_used_pitch': np.zeros((num_samples, 1)),
        'total_pitch_class_histogram': np.zeros((num_samples, 12)),
        'pitch_range': np.zeros((num_samples, 1)),
        'avg_pitch_shift': np.zeros((num_samples, 1))
    }

    metrics_list = list(set1_eval.keys())

    for i in range(0, num_samples):
        feature = core.extract_feature(set1[i])

        for metric in metrics_list:
            set1_eval[metric][i] = getattr(
                core.metrics(), metric
            )(feature)

    # repeat for second dataset
    set2_eval = {
        'total_used_pitch': np.zeros((num_samples, 1)),
        'total_pitch_class_histogram': np.zeros((num_samples, 12)),
        'pitch_range': np.zeros((num_samples, 1)),
        'avg_pitch_shift': np.zeros((num_samples, 1))
    }

    for i in range(0, num_samples):
        feature = core.extract_feature(set2[i])

        for metric in metrics_list:
            set2_eval[metric][i] = getattr(
                core.metrics(), metric
            )(feature)

    # statistic analysis: absolute measurement
    absolute_measurement = ""
    for i in range(0, len(metrics_list)):
        absolute_measurement += metrics_list[i] + ':'
        absolute_measurement += "\n" + '------------------------\n'
        absolute_measurement += "\n" + set1name
        absolute_measurement += "\n" + \
            '  mean: %s' % np.mean(set1_eval[metrics_list[i]], axis=0)
        absolute_measurement += "\n" + \
            '  std: %s' % np.std(set1_eval[metrics_list[i]], axis=0)

        absolute_measurement += "\n\n" + set2name
        absolute_measurement += "\n" + \
            '  mean: %s' % np.mean(set2_eval[metrics_list[i]], axis=0)
        absolute_measurement += "\n" + \
            '  std: %s\n\n' % np.std(set2_eval[metrics_list[i]], axis=0)

    with open(os.path.join(dstfolder, '1absolute_measurement.txt'), 'w') as f:
        f.writelines(absolute_measurement)

    # ## Relative measurement: generalizes the result among features with various dimensions
    #

    # the features are sum- marized to
    # - the intra-set distances
    # - the difference of intra-set and inter-set distances.

    # exhaustive cross-validation for intra-set distances measurement

    loo = LeaveOneOut()
    loo.get_n_splits(np.arange(num_samples))
    set1_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
    set2_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
    for i in range(len(metrics_list)):
        for train_index, test_index in loo.split(np.arange(num_samples)):
            set1_intra[test_index[0]][i] = utils.c_dist(
                set1_eval[metrics_list[i]][test_index], set1_eval[metrics_list[i]][train_index])
            set2_intra[test_index[0]][i] = utils.c_dist(
                set2_eval[metrics_list[i]][test_index], set2_eval[metrics_list[i]][train_index])

    # exhaustive cross-validation for inter-set distances measurement
    loo = LeaveOneOut()
    loo.get_n_splits(np.arange(num_samples))
    sets_inter = np.zeros((num_samples, len(metrics_list), num_samples))

    for i in range(len(metrics_list)):
        for train_index, test_index in loo.split(np.arange(num_samples)):
            sets_inter[test_index[0]][i] = utils.c_dist(
                set1_eval[metrics_list[i]][test_index], set2_eval[metrics_list[i]])

    # visualization of intra-set and inter-set distances
    plot_set1_intra = np.transpose(
        set1_intra, (1, 0, 2)).reshape(len(metrics_list), -1)
    plot_set2_intra = np.transpose(
        set2_intra, (1, 0, 2)).reshape(len(metrics_list), -1)
    plot_sets_inter = np.transpose(
        sets_inter, (1, 0, 2)).reshape(len(metrics_list), -1)
    for i in range(0, len(metrics_list)):
        sns.kdeplot(plot_set1_intra[i], label='intra %s' % set1name)
        sns.kdeplot(plot_sets_inter[i], label='inter')
        sns.kdeplot(plot_set2_intra[i], label='intra %s' % set2name)

        plt.title(metrics_list[i])
        plt.xlabel('Euclidean distance')
        plt.xlabel('Density')
        output.savefig(plt.gcf(), os.path.join(
            dstfolder, '3' + metrics_list[i] + '.png'))
        plt.clf()

    # the difference of intra-set and inter-set distances.
    distance_text = ''
    for i in range(0, len(metrics_list)):
        distance_text += metrics_list[i] + ':\n'
        distance_text += '------------------------\n' 
        distance_text += "\n" + set1name
        distance_text += "\n" + '  Kullback-Leibler divergence: %s' % utils.kl_dist(
            plot_set1_intra[i], plot_sets_inter[i])
        distance_text += "\n" + '  Overlap area: %s' % utils.overlap_area(
            plot_set1_intra[i], plot_sets_inter[i])

        distance_text += "\n" + set2name
        distance_text += "\n" + '  Kullback-Leibler divergence: %s' % utils.kl_dist(
            plot_set2_intra[i], plot_sets_inter[i])
        distance_text += "\n" + '  Overlap area: %s\n\n' % utils.overlap_area(
            plot_set2_intra[i], plot_sets_inter[i])

    with open(os.path.join(dstfolder, '4distance_text.txt'), 'w') as f:
        f.writelines(distance_text)


# Assign dataset path
if __name__ == "__main__":
    globstr1 = 'd:/thesis_code/model_fulldata1/samples/sessiontune45840_transposed/temp*.mid'
    set1 = glob.glob(
        globstr1)
    set1name = 'generated'
    print 'we have %s samples' % len(set1)
    num_samples = len(set1)

    globstr2 = 'd:/data/folkdataset/4_transposed_split_4bars/*.mid'
    set2 = glob.glob(globstr2)
    random.shuffle(set2)
    set2 = set2[:num_samples]

    set2name = 'training'

    dstfolder = 'comparison1'
    if not os.path.exists(dstfolder):
        os.mkdir(dstfolder)

    with open(os.path.join(dstfolder, '0info.txt'), 'w') as f:
        f.writelines("Comparison between %s and %s" %(globstr1, globstr2))

    main(set1, set2, set1name, set2name, dstfolder)
