# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source python -m egg.zoo.dsprites_bvae.train --lr=1e-3 --batch_size=128 --n_epochs=100 --vocab_size=2tree.

import torch
import torch.utils.data
from scipy.stats import spearmanr
from torchvision import datasets, transforms, utils
from torch import nn
from torch.nn import functional as F
import egg.core as core
import pathlib
from torch.autograd import Variable
import json
from typing import Union, Callable
import os



import numpy as np
from .archs import VisualReceiver, VisualSender
from egg.zoo.dsprites_bvae.data_loaders.data_loaders import get_dsprites_dataloader
from egg.core.language_analysis import TopographicSimilarity, PosDisent


class TopographicSimilarityLatents(TopographicSimilarity):
    def __init__(self,
                 sender_input_distance_fn: Union[str, Callable] = 'cosine',
                 message_distance_fn: Union[str, Callable] = 'edit',
                 compute_topsim_train_set: bool = False,
                 compute_topsim_test_set: bool = True):

        super().__init__(sender_input_distance_fn, message_distance_fn, compute_topsim_train_set, compute_topsim_test_set)

    def compute_similarity(self, sender_input: torch.Tensor, messages: torch.Tensor, mode: str,  epoch: int):
        def compute_distance(_list, distance):
            return [distance(el1, el2)
                    for i, el1 in enumerate(_list[:-1])
                    for j, el2 in enumerate(_list[i+1:])
                    ]

        messages = [msg.tolist() for msg in messages]
        input_dist = compute_distance(
            sender_input.numpy(), self.sender_input_distance_fn)
        message_dist = compute_distance(messages, self.message_distance_fn)
        topsim = spearmanr(input_dist, message_dist,
                           nan_policy='raise').correlation

        output_message = json.dumps(dict(topsim=topsim, epoch=epoch))
        print(output_message, flush=True)


def reconstruction_loss(x, x_recon, distribution='bernoulli'):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps

class betaVAE_Game(nn.Module):
    def __init__(self, sender, receiver, z_dim=10, beta=4):
        """Model proposed in the original beta-VAE paper(Higgins et al, ICLR, 2017)."""

        super().__init__()

        self.sender = sender
        self.receiver = receiver

        self.z_dim = z_dim
        self.beta = beta


    def forward(self, *batch):
        sender_input = batch[0]
        latent_values = batch[1]
        label = batch[2]

        distributions = self.sender(sender_input)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]

        if self.train:
            message = reparametrize(mu, logvar)
        else:
            message = mu

        receiver_output = self.receiver(message)

        recon_loss = reconstruction_loss(sender_input, receiver_output)
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

        beta_vae_loss = recon_loss + self.beta * total_kld


        log = core.Interaction(
            sender_input=label,
            receiver_input=None,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            labels=None,
            message_length=torch.ones(message.size(0)),
            aux={}
        )

        return beta_vae_loss.mean(), log


class ImageDumpCallback(core.Callback):
    def __init__(self, eval_dataset, image_shape=(64,64)):
        super().__init__()
        self.eval_dataset = eval_dataset
        self.image_shape = image_shape

    def on_epoch_end(self, loss, logs, epoch):
        dump_dir = pathlib.Path.cwd() / 'dump' / str(epoch)
        dump_dir.mkdir(exist_ok=True, parents=True)

        state = self.trainer.game.train
        self.trainer.game.eval()

        l = len(self.eval_dataset)

        for i in range(5):
            example_id = np.random.randint(0, l)
            example = self.eval_dataset[example_id]

            example = (example[0].unsqueeze(0), example[1].unsqueeze(0), example[2].unsqueeze(0))

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            example = core.move_to(example, device)
            _, interaction = self.trainer.game(*example)

            image = example[0][0]

            output = interaction.receiver_output.view(*self.image_shape)
            image = image.view(*self.image_shape)
            utils.save_image(
                torch.cat([image, output], dim=1), dump_dir / (str(i) + '.png'))
        self.trainer.game.train(state)


def main(params):
    opts = core.init(params=params)

    root = os.path.join('data', 'dsprites-dataset', 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    if not os.path.exists(root):
        import subprocess
        print('Now download dsprites-dataset')
        subprocess.call([os.path.join('egg', 'zoo', 'dsprites_bvae', 'data_loaders', 'download_dsprites.sh')])
        print('Finished')

    train_loader, test_loader = get_dsprites_dataloader(path_to_data=root,
                                                            batch_size=opts.batch_size, image=True)
    image_shape = (64, 64)





    sender = VisualSender()
    receiver = VisualReceiver()
    game = betaVAE_Game(sender, receiver)

    optimizer = core.build_optimizer(game.parameters())


    # initialize and launch the trainer
    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader, validation_data=test_loader,
                           callbacks=[core.ConsoleLogger(as_json=True, print_train_loss=True),
                                      ImageDumpCallback(test_loader.dataset, image_shape=image_shape),
                                      TopographicSimilarityLatents('euclidean', 'euclidean'), PosDisent()])
    trainer.train(n_epochs=opts.n_epochs)

    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])