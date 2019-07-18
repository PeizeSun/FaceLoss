# -*-coding:utf-8-*-

import imageio
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def visual_feature_space(features, labels, num_classes, epoch, acc, name_dict, mode='2D'):
    """Plot features on 2D plane and 3D plane.
    Args:
        features: (num_instances, num_features).
        labels: (num_instances).
    """
    # from IPython import embed; embed()
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    features = tsne.fit_transform(features[:500, :])
    labels = labels[:500]
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    if mode == '2D':
        for label_idx in range(num_classes):
            plt.scatter(
                features[labels == label_idx, 0],
                features[labels == label_idx, 1],
                c=colors[label_idx],
                s=10,
            )
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
        plt.title("epoch: %2d   accuracy: %.4f" % (epoch, acc))
        if not os.path.exists(name_dict):
            os.mkdir(name_dict)
        save_name = os.path.join(name_dict, 'epoch_' + str(epoch) + '.png')
        plt.savefig(save_name, bbox_inches='tight')
        plt.close()
    elif mode == '3D':
        fig = plt.figure()
        ax = Axes3D(fig)
        X, Y, Z = features[:, 0], features[:, 1], features[:, 2]
        for x, y, z, s in zip(X, Y, Z, labels):
            ax.text(x, y, z, s, color=colors['s'])
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_zlim(Z.min(), Z.max())
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
        plt.title("epoch: %2d   accuracy: %.4f" % (epoch, acc))
        if not os.path.exists(name_dict):
            os.mkdir(name_dict)
        save_name = os.path.join(name_dict, 'epoch_' + str(epoch) + '.png')
        plt.savefig(save_name, bbox_inches='tight')
        plt.close()


def visual_feature_space_record(vis_record, nr_sample_per_img=500, mode='2D'):

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    tsne = TSNE(n_components=2, init='pca', random_state=0)

    embedding_collection = list()
    for record_key in vis_record:
        embedding_collection.append(vis_record[record_key][0][:nr_sample_per_img])
    embedding_collection = np.concatenate(embedding_collection, 0)

    new_embedding_collection = tsne.fit_transform(embedding_collection)
    print('tsne caculation complete')
    axis_range = np.abs(new_embedding_collection).max()
    count = -1
    for record_key in vis_record:
        count += 1
        if record_key[:5] == 'train':
            name_dict = 'train/'
        elif record_key[:4] == 'test':
            name_dict = 'test/'
        _, labels, num_classes, epoch, acc = vis_record[record_key]
        features = new_embedding_collection[count * nr_sample_per_img: (count+1) * nr_sample_per_img]
        labels = labels[:nr_sample_per_img]
        if mode == '2D':
            for label_idx in range(num_classes):
                plt.scatter(
                    features[labels == label_idx, 0],
                    features[labels == label_idx, 1],
                    c=colors[label_idx],
                    s=20,
                )
            plt.xlim((-axis_range, axis_range))
            plt.ylim((-axis_range, axis_range))
            plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
            plt.title("epoch: %2d   accuracy: %.4f" % (epoch, acc))
            if not os.path.exists(name_dict):
                os.mkdir(name_dict)
            save_name = os.path.join(name_dict, 'epoch_' + str(epoch) + '.png')
            plt.savefig(save_name, bbox_inches='tight')
            plt.close()
        elif mode == '3D':
            fig = plt.figure()
            ax = Axes3D(fig)
            X, Y, Z = features[:, 0], features[:, 1], features[:, 2]
            for x, y, z, s in zip(X, Y, Z, labels):
                ax.text(x, y, z, s, color=colors['s'])
            ax.set_xlim(X.min(), X.max())
            ax.set_ylim(Y.min(), Y.max())
            ax.set_zlim(Z.min(), Z.max())
            plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
            plt.title("epoch: %2d   accuracy: %.4f" % (epoch, acc))
            if not os.path.exists(name_dict):
                os.mkdir(name_dict)
            save_name = os.path.join(name_dict, 'epoch_' + str(epoch) + '.png')
            plt.savefig(save_name, bbox_inches='tight')
            plt.close()


def create_gif(gif_name, filepath, duration=0.1):
    """
    :param gif_name:
    :param filepath:
    :param duration:
    :return:
    """
    frames = []
    file_list = os.listdir(filepath)
    for f in file_list:
        head, tail = os.path.splitext(f)
        if tail != '.png':
            file_list.remove(f)
    file_list.sort()
    image_list = [os.path.join(filepath, x) for x in file_list]
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)


def main():
    gif_name = {
        'train': 'features_train.gif',
        'test' : 'features_test.gif'
    }

    filepath = {'train': 'train/',
                'test': 'test/'}

    for phase in ['train', 'test']:
        create_gif(gif_name[phase], filepath[phase], duration=0.5)


if __name__ == '__main__':
    main()