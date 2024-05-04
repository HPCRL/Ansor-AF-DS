import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches
from matplotlib.legend_handler import HandlerTuple


def drawline():
    # relative speedup
    m1_t = pd.DataFrame({

        'Ansor': [
8494.286372/1000
, 6083.646365/1000
,1925.988005/1000
,3956.202227/1000
,1198.232836/1000
],
        'cuDNN': [
6229.621622/1000
, 4891.905213/1000
,1271.284244/1000
,2998.06805/1000
,669.8506466/1000
,],
    })


    fig, ax1 = plt.subplots()
    ax1.spines["top"].set_visible(True)
    ax1.spines["right"].set_visible(True)
    ax1.spines["bottom"].set_visible(True)
    ax1.spines["left"].set_visible(True)
    x_label = [ '224x3x64_7*', '56x64x64_3', '28x512x256_1*', '14x256x256_3', '14x1024x512_1*']
    x = np.arange(len(x_label))

    hatch_dict = {0: '', 1: '', 2: 'xx', 3: '++', 4: '.'}
    labels = ['Ansor', 'cuDNN']
    colors = ['#117733', '#88CCEE', 'orange', 'dodgerblue', 'lightgreen']
    width = 0.2
    # rects1 = ax1.bar(x - 2* width, m1_t['ansor-100'], width, label=labels[0], hatch=hatch_dict[0], alpha=.99,
    #                  color=colors[0])
    rects2 = ax1.bar(x, m1_t['Ansor'], width, label=labels[0], hatch=hatch_dict[0], alpha=.99,
                     color=colors[0])
    rects3 = ax1.bar(x + width, m1_t['cuDNN'], width, label=labels[1], hatch=hatch_dict[1], alpha=.99,
                     color=colors[1])
    # rects4 = ax1.bar(x + width, m1_t['t20'], width, label=labels[3], hatch=hatch_dict[3], alpha=.99,
    #                  color=colors[3])
    # rects5 = ax1.bar(x + 2*width, m1_t['our'], width, label=labels[4], hatch=hatch_dict[4], alpha=.99,
    #                  color=colors[4])

    # plt.axhline(y=1, color='black', linestyle='--')



    ax1.tick_params(axis='y', labelsize=20)
    
    ax1.set_yticks(np.arange(0, 11, 2)) 
    # ax1.set_xticklabels(x_label)
    plt.xticks(x + width / 4, x_label)

    plt.tick_params(axis='x', which='major', labelsize=16)
    plt.xticks(rotation=0)

    # markers = ['o-', '^-', 'x-']
    # ax2 = ax1.twinx()
    # ax2.plot(x_label, m2_t['tvm_time'], markers[0], label='tvm_time', color="blue")
    # ax2.tick_params(axis='y', labelsize=16, labelcolor='blue')
    # ax2.set_ylim(0, 0.55)
    # ax2.set_yticks(ax2.get_yticks()[-3:])

    # ax1.yaxis.grid(linestyle='--', linewidth='0.5')
    ax1.set_ylabel('TFlops', size=20)
    # ax2.set_ylabel('TVM execution time', size=18, )
    # ax2.yaxis.label.set_color('blue')

    # def autolabel(rects, vals):
    #     """Attach a text label above each bar in *rects*, displaying its height."""
    #     i = 0
    #     for rect in rects:
    #         height = rect.get_height()
    #         val = vals[i]
    #         i = i + 1
    #         ax1.annotate('{:4.2f}'.format(val),
    #                      xy=(rect.get_x() + 0.1, height),
    #                      xytext=(0, 20),  # 3 points vertical offset
    #                      textcoords="offset points",
    #                      ha='right', va='center', size=13, fontweight='bold', rotation=-90)

    # abs = abs / 1000
    # autolabel(rects1, abs['abs-our'])
    # autolabel(rects4, abs['abs-ansor'])

    leg_artists = []
    for i in range(len(hatch_dict)):
        p = matplotlib.patches.Patch(facecolor=colors[i], hatch=hatch_dict[i], alpha=.99)
        # linem = plt.plot([], [], markers[i], markerfacecolor=colors[i], markeredgecolor=colors[i], color=colors[i])[0]
        leg_artists.append((p))

    # ax1.legend(leg_artists, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=5,
    #            handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=18)
    ax1.text(0.89, 0.70, 'RTX3090', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,
             fontsize=20)

    # tt = 15.7
    # textstr = ('machine peak = %.1f TFLOPS' % (tt))
    #
    # # place a text box in upper left in axes coords
    # props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    # ax1.text(0.03, 0.95, textstr, transform=ax1.transAxes, fontsize=18,
    #          verticalalignment='top', bbox=props)

    fig.set_figheight(3)
    fig.set_figwidth(11)
    fig.tight_layout()

    plt.show()

    fig.savefig('cudnn-intro3090.pdf')


if __name__ == "__main__":
    # execute only if run as a script
    drawline()