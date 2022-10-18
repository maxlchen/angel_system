import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams.update({'font.size': 12})
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea, AnchoredText
import numpy as np
import PIL
from pathlib import Path
import os

from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc

import logging


log = logging.getLogger("ptg_eval")

class EvalVisualization:
    def __init__(self, labels, gt_true_mask, dets_per_valid_time_w, output_dir=''):
        """
        :param labels: Array of class labels (str)
        :param gt_true_pos_mask: Matrix of size (number of valid time windows x number classes) where True
            indicates a true class example, False inidcates a false class example
        :param dets_per_time_w: Matrix of size (number of valid time windows x number classes) filled with 
            the max confidence score per class for any detections in the time window
        :param output_dir: Directory to write the plots to
        """
        self.output_dir = Path(os.path.join(output_dir, "plots/"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = str(self.output_dir)

        self.labels = labels

        self.gt_true_mask = gt_true_mask
        self.dets_per_valid_time_w = dets_per_valid_time_w

    def plot_pr_curve(self):
        """
        Plot the PR curve for each label
        """
        pr_plot_dir = Path(os.path.join(self.output_dir, "pr"))
        pr_plot_dir.mkdir(parents=True, exist_ok=True)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.labels)))

        all_y_true = []
        all_s = []
        # ============================
        # Get PR plot per class 
        # ============================
        precision = dict()
        recall = dict()
        average_precision = dict()
        for id, label in enumerate(self.labels):
            class_dets_per_time_w = self.dets_per_valid_time_w[:, id]
            mask_per_class = self.gt_true_mask[:, id]

            ts = class_dets_per_time_w[mask_per_class]
            fs = class_dets_per_time_w[~mask_per_class]

            s = np.hstack([ts, fs]).T
            y_true = np.hstack([np.ones(len(ts), dtype=bool),
                     np.zeros(len(fs), dtype=bool)]).T
            s.shape = (-1, 1)
            y_true.shape = (-1, 1)

            all_y_true.extend(y_true)
            all_s.extend(s)

            precision[id], recall[id], _ = precision_recall_curve(y_true, s)
            average_precision[id] = average_precision_score(y_true, s)

        # ============================
        # Average values
        # ============================
        precision["macro"], recall["macro"], _ = precision_recall_curve(all_y_true, all_s)
        average_precision["macro"] = average_precision_score(all_y_true, all_s, average="macro")

        # ============================
        # Plot
        # ============================
        for id, label in enumerate(self.labels):
            # ============================
            # Setup figure
            # ============================
            fig, ax = plt.subplots(figsize=(14, 8))

            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_title("Precision vs. Recall")
            
            # ============================
            # Add F1 score 
            # ============================
            fscores = np.linspace(0.2, 0.8, num=4)
            for f_score in fscores:
                x = np.linspace(0.001, 1)
                y = f_score * x / (2 * x - f_score)
                (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2, linestyle='dashed')
                plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

            # plot average values
            av_display = PrecisionRecallDisplay(
                recall=recall["macro"],
                precision=precision["macro"],
                average_precision=average_precision["macro"],
            )
            av_display.plot(ax=ax, name="Macro-averaged over all classes", 
                            color="navy", linestyle=":", linewidth=4)

            # plot class values
            display = PrecisionRecallDisplay(
                recall=recall[id],
                precision=precision[id],
                average_precision=average_precision[id],
            )
            display.plot(ax=ax, name=label, color=colors[id])

            # ============================
            # Save
            # ============================
            # Add legend and f1 curves to plot
            handles, labels = ax.get_legend_handles_labels()
            handles.extend([l])
            labels.extend(["iso-f1 curves"])
            ax.legend(handles=handles, labels=labels, loc="best")

            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")

            fig.savefig(f"{pr_plot_dir}/{label.replace(' ', '_')}.png")
            plt.close(fig)

    def plot_roc_curve(self):
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.labels)))

        roc_plot_dir = Path(os.path.join(self.output_dir, "roc"))
        roc_plot_dir.mkdir(parents=True, exist_ok=True)

        # ============================
        # Get ROC and AUC per class 
        # ============================
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for id, label in enumerate(self.labels):
            class_dets_per_time_w = self.dets_per_valid_time_w[:, id]
            mask_per_class = self.gt_true_mask[:, id]

            ts = class_dets_per_time_w[mask_per_class]
            fs = class_dets_per_time_w[~mask_per_class]

            s = np.hstack([ts, fs]).T
            y_true = np.hstack([np.ones(len(ts), dtype=bool),
                     np.zeros(len(fs), dtype=bool)]).T
            s.shape = (-1, 1)
            y_true.shape = (-1, 1)

            fpr[id], tpr[id], _ = roc_curve(y_true, s)
            roc_auc[id] = auc(fpr[id], tpr[id])

        # ============================
        # Average values
        # ============================
        n_classes = len(self.labels)

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # ============================
        # Plot
        # ============================
        for id, label in enumerate(self.labels):
            # ============================
            # Setup figure
            # ============================
            fig, ax = plt.subplots(figsize=(14, 8))

            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_title("Receiver Operating Characteristic (ROC)")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")

            # plot average values
            ax.plot(
                fpr["macro"],
                tpr["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )

            # plot class values
            ax.plot(fpr[id], tpr[id], color=colors[id], lw=2,
                    label=label + " (area = {0:0.2f})".format(roc_auc[id]))

            # ============================
            # Save
            # ============================
            ax.legend(loc="best")
            fig.savefig(f"{roc_plot_dir}/{label.replace(' ', '_')}.png")
            plt.close(fig)
        
def plot_activities_confidence(labels, gt, dets, custom_range=None, output_dir='', custom_range_color="red"):
    """
    Plot activity confidences over time
    :param gt: 
    :param dets: 
    
    :param custom_range: Optional tuple indicating the starting and ending times of an additional
                            range to highlight in addition to the `gt_ranges`.
    :param custom_range_color: The color of the additional range to be drawn. If not set, we will
                                use "red".
    """
    for label in labels:
        gt_ranges = gt.loc[gt['class'] == label]
        det_ranges = dets.loc[dets['class'] == label]

        if not gt_ranges.empty and not det_ranges.empty:
            # ============================
            # Setup figure
            # ============================
            # Determine time range to plot
            min_start_time = min(gt_ranges['start'].min(), det_ranges['start'].min())
            max_end_time = max(gt_ranges['end'].max(), det_ranges['end'].max())
            total_time_delta = max_end_time - min_start_time
            pad = 0.05 * total_time_delta

            # Setup figure
            fig = plt.figure(figsize=(14, 6))
            ax = fig.add_subplot(111)
            ax.set_title(f"Window Confidence over time for \"{label}\"\n in time range {min_start_time}:{max_end_time}")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Confidence")
            ax.set_ylim(0, 1.05)
            ax.set_xlim(0-pad, (max_end_time - min_start_time + pad))

            # ============================
            # Ground truth
            # ============================
            # Bar plt to show bars where the "true" time ranges are for the activity.
            xs_bars = [p - min_start_time for p in gt_ranges['start'].tolist()]
            ys_gt_regions = [1] * gt_ranges.shape[0]
            bar_widths = [p for p in gt_ranges['end']-gt_ranges['start']]
            ax.bar(xs_bars, ys_gt_regions, width=bar_widths, align="edge", color="lightgreen", label="Ground truth")

            if custom_range:
                assert len(custom_range) == 2, "Assuming only two float values for custom range"
                xs_bars2 = [custom_range[0]]
                ys_height = [1.025] #[0.1]
                bar_widths2 = [custom_range[1]-custom_range[0]]
                ys_bottom = [0] #[1.01]
                # TODO: Make this something that is added by clicking?
                ax.bar(xs_bars2, ys_height,
                    width=bar_widths2, bottom=ys_bottom, align="edge",
                    color=custom_range_color, alpha=0.5)

            # ============================
            # Detections
            # ============================
            xs2_bars = [p - min_start_time for p in det_ranges['start'].tolist()]
            ys2_det_regions = [p for p in det_ranges['conf']]
            bar_widths2 = [p for p in det_ranges['end']-det_ranges['start']]
            ax.bar(xs2_bars, ys2_det_regions, width=bar_widths2, align="edge", edgecolor="blue", fill=False, label="Detections")

            ax.legend(loc="upper right")
            ax.plot

            # ============================
            # Save
            # ============================
            #plt.show()
            activity_plot_dir = Path(os.path.join(output_dir, "activities"))
            activity_plot_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(f"{str(activity_plot_dir)}/{label.replace(' ', '_')}.png")
            plt.close(fig)
        else:
            log.warning(f"No detections/gt found for \"{label}\"")
