'''
visualize video vector by tsne
'''
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle

FEATURE_SAVEPATH = "data/feature/goal_miss_sub"
FIGURE_SAVEPATH = FEATURE_SAVEPATH.replace("feature", "figure_tsne")
FEATURE_DIM = 512
CLASS_LIST = ['goal-kick', 'miss-kick', 'miss-basket',
              'goal-basket', 'goal-hand', 'miss-hand']
'''CLASS_LIST = ['come-kick', 'go-kick', 'come-pitch',
              'go-pitch', 'come-run', 'go-run']
CLASS_LIST = ['build-pc', 'build-bic', 'dis-eng',
              'build-eng', 'dis-pc', 'dis-bic',
              'build-lego', 'dis-lego', 'dis-sma',
              'build-sma'
              ]
'''

def visualize():
    with open(FEATURE_SAVEPATH, 'rb') as f:
        feature_dict = pickle.load(f)
    feature_np = np.zeros((len(feature_dict), FEATURE_DIM))
    label_list = []
    label_total_list = [0.] * len(CLASS_LIST)
    for idx, (k, v) in enumerate(feature_dict.items()):
        label = int(k.split("___")[-1])
        label_list.append(label)
        feature_np[idx] = v

    label_np = np.array(label_list)

    tsne = TSNE()
    tsne_trans = tsne.fit_transform(feature_np)
    cmap = plt.get_cmap("Set1")
    plt.figure()
    for idx, class_name in enumerate(CLASS_LIST):
        feature_bool = label_np == idx
        feature_trans = tsne_trans[feature_bool, :]
        print(feature_trans.shape)
        plt.scatter(feature_trans[:,0], feature_trans[:,1], s=3100, color=cmap(idx), marker=f"${class_name}$")
    plt.savefig(FIGURE_SAVEPATH)
    plt.show()
    print("saved", FIGURE_SAVEPATH)

if __name__ == "__main__":
    visualize()
