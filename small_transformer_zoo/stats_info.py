import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

dataset1 = pd.read_csv('metadata/mnist.csv')
dataset2 = pd.read_csv('metadata/ag_news.csv')

dataset1['test_top1_accuracy'] = dataset1['test_top1_accuracy']
dataset2['test_top1_accuracy'] = dataset2['test_accuracy']

dataset1['ckpt_epoch'] = dataset1['ckpt_epoch'].apply(lambda x: "best" if x == "best1" else x)

def get_accuracies(data):
    epochs = ["50", "75", "100", "best"]
    return [data[data['ckpt_epoch'] == epoch]['test_top1_accuracy'] if not data[data['ckpt_epoch'] == epoch].empty else None for epoch in epochs]

accuracies1 = get_accuracies(dataset1)
accuracies2 = get_accuracies(dataset2)

plt.rcParams.update({'font.size': 17})

fig, axs = plt.subplots(2, 4, figsize=(22, 10), sharex='col', sharey='row')

colors = sns.color_palette("husl", 2)

def plot_histogram(ax, data, color, label):
    if data is not None:
        sns.histplot(data, kde=False, ax=ax, color=color, bins=25, stat='count', binrange=(0, 1), discrete=False, edgecolor='white', linewidth=0.5, label=label)
        ax.set_yscale('log')
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0.1)

epochs = ["Epoch 50", "Epoch 75", "Epoch 100", "Best"]

for i, accuracy in enumerate(accuracies1):
    plot_histogram(axs[0, i], accuracy, colors[0], 'MNIST')
    axs[0, i].set_title(epochs[i], fontsize=16)

for i, accuracy in enumerate(accuracies2):
    plot_histogram(axs[1, i], accuracy, colors[1], 'AGNews')
    
# After plotting, ensure y-axis ticks are appropriate for log scale
for ax in axs.flat:
    ax.yaxis.set_major_locator(plt.LogLocator(base=10))
    ax.yaxis.set_minor_locator(plt.LogLocator(base=10, subs='all'))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())

for ax in axs[1, :]:
    ax.set_xlabel('Accuracy')

for ax in axs.flat:
    ax.set_facecolor('#f0f0f0')
    ax.grid(True, color='white', linestyle='-', linewidth=1)
    ax.set_axisbelow(True)

axs[0, 0].set_ylabel('Number of samples', fontsize=17)
axs[1, 0].set_ylabel('Number of samples', fontsize=17)

# Get handles and labels from both MNIST and AGNews plots
handles1, labels1 = axs[0, 0].get_legend_handles_labels()
handles2, labels2 = axs[1, 0].get_legend_handles_labels()

# Combine handles and labels
handles = handles1 + handles2
labels = labels1 + labels2

# Create legend
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=16)

plt.tight_layout()
plt.subplots_adjust(hspace=0.2, wspace=0.1, top=0.9)  # Adjusted top to make room for legend

plt.savefig('dataset_accuracy_histograms.pdf', bbox_inches='tight', dpi=300)
plt.close()
