import torch

num_feat = 16520
num_class = 5

def plot_llk(train_elbo, test_elbo):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy as sp
    import seaborn as sns
    plt.figure(figsize=(30, num_class))
    sns.set_style("whitegrid")
    data = np.concatenate([np.arange(len(test_elbo))[:, sp.newaxis], -test_elbo[:, sp.newaxis]], axis=1)
    df = pd.DataFrame(data=data, columns=['Training Epoch', 'Test ELBO'])
    g = sns.FacetGrid(df, size=num_class, aspect=1.5)
    g.map(plt.scatter, "Training Epoch", "Test ELBO")
    g.map(plt.plot, "Training Epoch", "Test ELBO")
    plt.savefig('./vae_results/test_elbo_vae.png')
    plt.close('all')


def nhanes_test_tsne(vae=None, test_loader=None):
    """
    This is used to generate a t-sne embedding of the vae
    """
    name = 'VAE'
    data = torch.from_numpy(test_loader.dataset.test_data).float()
    nhanes_labels = torch.from_numpy(test_loader.dataset.test_labels).float()
    z_loc, z_scale = vae.encoder(data)
    plot_tsne(z_loc, nhanes_labels, name)


def nhanes_test_tsne_ssvae(name=None, ssvae=None, test_loader=None):
    """
    This is used to generate a t-sne embedding of the ss-vae
    """
    if name is None:
        name = 'SS-VAE'
    data = torch.from_numpy(test_loader.dataset.test_data).float()
    print(data.size())
    nhanes_labels = torch.from_numpy(test_loader.dataset.test_labels).float()
    z_loc, z_scale = ssvae.encoder_z([data, nhanes_labels])
    plot_tsne(z_loc, nhanes_labels, name)


def plot_tsne(z_loc, classes, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE
    model_tsne = TSNE(n_components=2, random_state=0)
    z_states = z_loc.detach().cpu().numpy()
    z_embed = model_tsne.fit_transform(z_states)
    classes = classes.detach().cpu().numpy()
    fig = plt.figure()
    for ic in range(num_class):
        ind_vec = np.zeros_like(classes)
        ind_vec[:, ic] = 1
        ind_class = classes[:, ic] == 1
        color = plt.cm.Set1(ic)
        plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=num_class, color=color)
        plt.title("Latent Variable T-SNE per Class")
        fig.savefig('./vae_results/'+str(name)+'_embedding_'+str(ic)+'.png')
    fig.savefig('./vae_results/'+str(name)+'_embedding.png')
