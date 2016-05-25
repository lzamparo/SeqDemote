#!/usr/bin/env python
from optparse import OptionParser
import glob, os

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
sns_colors = sns.color_palette('deep')

################################################################################
# plot_roc.py
#
# Plot ROCs for all targets, including a composite curve comparing a few.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <roc_dir>'
    parser = OptionParser(usage)
    parser.add_option('-t', dest='targets_file')
    (options,args) = parser.parse_args()

    if len(args) != 1:
        parser.error('Must provide ROC points file')
    else:
        roc_dir = args[0]

    # read target labels
    if options.targets_file:
        target_labels = [line.split()[0] for line in open(options.targets_file)]
    else:
        target_labels = ['Target %d'%(ti+1) for ti in range(len(glob.glob('%s/roc*.txt'%roc_dir)))]

    #######################################################
    # make all ROC plots
    #######################################################
    target_fpr = []
    target_tpr = []

    for roc_file in glob.glob('%s/roc*.txt' % roc_dir):
        ti = int(roc_file[roc_file.find('roc')+3:-4]) - 1

        target_fpr.append([])
        target_tpr.append([])
        for line in open(roc_file):
            a = line.split()
            target_fpr[-1].append(float(a[0]))
            target_tpr[-1].append(float(a[1]))

        plt.figure(figsize=(6,6))

        plt.scatter(target_fpr[-1], target_tpr[-1], s=8, linewidths=0, c=sns_colors[0])

        plt.title(target_labels[ti])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.grid(True)
        plt.tight_layout()

        out_pdf = '%s.pdf' % os.path.splitext(roc_file)[0]
        plt.savefig(out_pdf)
        plt.close()

    #######################################################
    # multiple ROC curve plot
    #######################################################
    # read AUCs
    target_aucs = [float(line.split()[1]) for line in open('%s/aucs.txt'%roc_dir)]

    # choose cells
    auc_targets = [(target_aucs[ti],ti) for ti in range(len(target_aucs))]
    auc_targets.sort()

    fig_quants = [0.05, .33, 0.5, .67, .95]
    auc_target_quants = quantile(auc_targets, fig_quants)

    # plot
    sns.set(style='white', font_scale=1.2)
    plt.figure(figsize=(6,6))

    si = 0
    for auc, ti in auc_target_quants:
        target_label = '%-9s AUC: %.3f' % (target_labels[ti], target_aucs[ti])
        plt.plot(target_fpr[ti], target_tpr[ti], c=sns_colors[si], label=target_label, linewidth=2.5, alpha=0.8)
        si += 1

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')

    plt.xlim((0,1))
    plt.ylim((0,1))

    ax = plt.gca()
    ax.xaxis.label.set_fontsize(17)
    ax.yaxis.label.set_fontsize(17)

    map(lambda xl: xl.set_fontsize(13), ax.get_xticklabels())
    map(lambda yl: yl.set_fontsize(13), ax.get_yticklabels())

    ax.grid(True, linestyle=':')
    plt.tight_layout()

    matplotlib.rcParams.update({'font.family': 'monospace'})
    plt.legend(loc=4, fontsize=12)

    plt.savefig('%s/range.pdf'%roc_dir)
    plt.close()


def quantile(ls, q):
    ''' Return the value at the quantile given. '''
    sls = sorted(ls)

    if type(q) == list:
        qval = []
        for j in range(len(q)):
            qi = int((len(sls)-1)*q[j])
            qval.append(sls[qi])
    else:
        qi = int(len(sls)*q)
        qval = sls[qi]

    return qval


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
    #pdb.runcall(main)
