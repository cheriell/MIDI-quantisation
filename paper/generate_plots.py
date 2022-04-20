import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='dark')
from pathlib import Path
import sys

figures_dir = Path(sys.path[0], 'figures/')
Path.mkdir(figures_dir, exist_ok=True)

def main():
    print('Generate plots')

    input_encoding_comparison_plot()

def input_encoding_comparison_plot():
    print('Input encoding comparison plot')

    # ========= data =========
    f1s = [
        [87.8, 87.3],
        [70.0, 87.8, 79.3, 91.3],
        [91.3, 70.0]
    ]  # f1-scores for pitch, onset, duration encodings
    labels = [
        ['midi', 'chroma'],
        ['absolute-raw', 'shift-raw', 'absolute-onehot', 'shift-onehot'],
        ['raw', 'onehot']
    ]
    groups = ['pitch', 'onset', 'duration']

    # ========= plot data =========
    # bar locations and size
    width = 0.25
    xs = [
        [width, width * 2],
        [width * 4.5, width * 5.5, width * 6.5, width * 7.5],
        [width * 10, width * 11]
    ]  # left border
    colors = ['Blues', 'Greens', 'Reds']

    # ========== plot ==========
    fig, ax = plt.subplots(figsize=(8, 6))

    # plot bars with values
    rects = []
    for i, group in enumerate(groups):
        rects.append([])
        for j in range(len(f1s[i])):
            color = plt.get_cmap(colors[i])((j+1)/len(f1s[i]))
            rect = ax.bar(xs[i][j], f1s[i][j], width, label=labels[i][j], color=color)
            rects[i].append(rect)

    # legends
    ax.legend(loc='lower right')
    for i, group in enumerate(groups):
        for j, rect in enumerate(rects[i]):
            ax.bar_label(rect)

    # x-axis and y-axis labels
    ax.set_xticks([(x[0] + x[-1]) / 2 for x in xs])
    ax.set_xticklabels(groups)
    plt.ylim([50, 100])
    ax.set_ylabel('F1-score (%)')
    
    # decorate
    plt.grid(axis='y')
    fig.tight_layout()

    plt.savefig(str(Path(figures_dir, 'input_encoding_comparison.png')))
    plt.savefig(str(Path(figures_dir, 'input_encoding_comparison.pdf')))

if __name__ == '__main__':
    main()