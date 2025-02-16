import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import json
import os
from tqdm import tqdm, trange

from models_train.layers.summarizer_transformer import Summarizer
from models_train.layers.discriminator import Discriminator
from models_train.utils import TensorboardWriter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

original_label = torch.tensor(1.0).to(device=device)
summary_label = torch.tensor(0.0).to(device=device)

class Solver(object):
    def __init__(self, config=None, train_loader=None, test_loader=None,margin=1.0):
        """Class that Builds, Trains and Evaluates SUM-GAN model"""
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.n_epoch_trained = 0
        self.margin = margin

    def build(self):

        # Build Modules
        self.linear_compress = nn.Linear(self.config.input_size, 
                                         self.config.hidden_size).to(device=device)

        self.summarizer = Summarizer(input_size=self.config.hidden_size, 
                                     hidden_size=self.config.hidden_size, 
                                     num_layers=self.config.num_layers).to(device=device)
        
        self.discriminator = Discriminator(input_size=self.config.hidden_size, 
                                           hidden_size=self.config.hidden_size, 
                                           num_layers=self.config.num_layers).to(device=device)
        
        self.model = nn.ModuleList([self.linear_compress,
                                     self.summarizer, 
                                     self.discriminator])

        if self.config.mode == 'train':
            # Build Optimizers
            self.s_e_optimizer = optim.Adam(list(self.summarizer.attn.parameters()) +
                                             list(self.summarizer.vae.e_lstm.parameters())
                + list(self.linear_compress.parameters()),
                lr=self.config.lr)

            self.d_optimizer = optim.Adam(list(self.summarizer.vae.d_lstm.parameters()) + 
                                          list(self.linear_compress.parameters()),
                lr=self.config.lr)

            self.c_optimizer = optim.Adam(
                list(self.discriminator.parameters())
                + list(self.linear_compress.parameters()),
                lr=self.config.discriminator_lr)

            self.model.train()
            # self.model.apply(apply_weight_norm)


            # Tensorboard
            self.writer = TensorboardWriter(self.config.log_dir)

    def loadfrom_checkpoint(self, path):
        '''
        Load state_dict from a pretrained model
        '''

        checkpoint = torch.load(path, map_location=device)
        self.n_epoch_trained = checkpoint['n_epoch_trained']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.config.mode == 'train':
            self.e_optimizer.load_state_dict(checkpoint['e_optimizer_state_dict'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            self.c_optimizer.load_state_dict(checkpoint['c_optimizer_state_dict'])
    
    @staticmethod
    def freeze_model(module):
        for p in module.parameters():
            p.requires_grad = False

    def reconstruction_loss(self, h_origin, h_fake,log_variance):
        """L2 loss between original-regenerated features at cLSTM's last hidden layer"""

        #return torch.norm(h_origin - h_fake, p=2)
        return torch.norm(h_origin - h_fake, p=2) 

    def prior_loss(self, mu, log_variance):
        """KL( q(e|x) || N(0,1) )"""
        return 0.5 * torch.sum(-1 + log_variance.exp() + mu.pow(2) - log_variance)

    def sparsity_loss(self, scores):
        """Summary-Length Regularization"""

        return torch.norm(torch.mean(scores) - self.config.summary_rate)

    def gan_loss(self, original_prob, fake_prob, uniform_prob):
        """Typical GAN loss + Classify uniformly scored features"""

        gan_loss = torch.mean(torch.log(original_prob) + torch.log(1 - fake_prob)
                              + torch.log(1 - uniform_prob))  # Discriminate uniform score

        return gan_loss
    
    ## Standard Contrastive Loss
    #def contrastive_loss(self, positive_pairs, negative_pairs):
        """Compute contrastive loss for the positive and negative pairs."""
    
    # Contrastive Loss
    def contrastive_loss(self, positive_pairs, negative_pairs, margin=1.0, lambdac=1.0, adaptive_weighting=False):
        """Compute contrastive loss for the positive and negative pairs with optional adaptive weighting."""
        
        # Calculate distances for positive pairs
        pos_distance = torch.norm(positive_pairs[0] - positive_pairs[1], p=2)  # L2 distance
        neg_distance = torch.norm(negative_pairs[0] - negative_pairs[1], p=2)

        # Compute the base contrastive loss
        loss = (1 - torch.exp(-pos_distance)) + torch.relu(neg_distance - margin)

        if adaptive_weighting:
            # adaptive weighting based on the relative magnitudes of distances
            total_distance = pos_distance + neg_distance
            adaptive_weight = pos_distance / (total_distance + 1e-8) # Adding epsilon (ϵ) to the denominator can improve stability:
            loss *= adaptive_weight  # Scale the loss by adaptive weight

        # Apply the lambda coefficient
        return lambdac * loss.mean()

    
    def train(self):
        step = 0
        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            s_e_loss_history = []
            d_loss_history = []
            c_loss_history = []
            for batch_i, image_features in enumerate(tqdm(self.train_loader, desc='Batch', ncols=80, leave=False)):
                print(batch_i)
                image_features=image_features[0]
                if image_features.size(1) > 10000:
                    continue

                # [batch_size=1, seq_len, 1024]
                # [seq_len, 1024]
                image_features = image_features.view(-1, self.config.input_size)
                
                # [seq_len, 1024]
                image_features_ = Variable(image_features).to(device=device)
                print('IMAGE FEATURES', len(image_features[0]))
                #---- Train sLSTM, eLSTM ----#
                if self.config.verbose:
                    tqdm.write('\nTraining sLSTM and eLSTM...')

                # [seq_len, 1, hidden_size]
                original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)

                scores, h_mu, h_log_variance, generated_features = self.summarizer(original_features)

                _, _, _, uniform_features = self.summarizer(original_features, uniform=True)

                # Discriminator output
                h_origin, original_prob = self.discriminator(original_features)
                h_fake, fake_prob = self.discriminator(generated_features)
                h_uniform, uniform_prob = self.discriminator(uniform_features)

                # Calculate reconstruction and GAN losses
                tqdm.write(f'original_p: {original_prob.item():.3f}, fake_p: {fake_prob.item():.3f}, uniform_p: {uniform_prob.item():.3f}')
                #tqdm.write(f'original_p: {original_prob.item():.3f}, summary_p: {sum_prob.item():.3f}')
                reconstruction_loss = self.reconstruction_loss(h_origin, h_fake,h_log_variance)
                prior_loss = self.prior_loss(h_mu, h_log_variance)
                sparsity_loss = self.sparsity_loss(scores)

                tqdm.write(
                    f'recon loss {reconstruction_loss.item():.3f}, prior loss: {prior_loss.item():.3f}, sparsity loss: {sparsity_loss.item():.3f}')

                s_e_loss = reconstruction_loss + prior_loss + sparsity_loss

                self.s_e_optimizer.zero_grad()
                s_e_loss.backward()  # retain_graph=True)
                
                # Gradient cliping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.s_e_optimizer.step()

                s_e_loss_history.append(s_e_loss.data)

                #---- Train dLSTM ----#
                if self.config.verbose:
                    tqdm.write('Training dLSTM...')

                # [seq_len, 1, hidden_size]
                original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)

                scores, h_mu, h_log_variance, generated_features = self.summarizer( original_features)
                _, _, _, uniform_features = self.summarizer(original_features, uniform=True)

                h_origin, original_prob = self.discriminator(original_features)
                h_fake, fake_prob = self.discriminator(generated_features)
                h_uniform, uniform_prob = self.discriminator(uniform_features)

                tqdm.write(f'original_p: {original_prob.item():.3f}, fake_p: {fake_prob.item():.3f}, uniform_p: {uniform_prob.item():.3f}')

                reconstruction_loss = self.reconstruction_loss(h_origin, h_fake,h_log_variance)
                gan_loss = self.gan_loss(original_prob, fake_prob, uniform_prob)

                tqdm.write(f'recon loss {reconstruction_loss.item():.3f}, gan loss: {gan_loss.item():.3f}')

            
                # Calculate contrastive loss
                pos_pairs = (h_origin, h_fake)  # Positive pairs
                neg_pairs = (h_fake, h_uniform)  # Negative pairs (fake vs. uniform)
                contrastive_loss_value = self.contrastive_loss(pos_pairs, neg_pairs, adaptive_weighting=True)

                
                d_loss = reconstruction_loss + gan_loss + contrastive_loss_value
                
                tqdm.write(f'contrastive loss {contrastive_loss_value.item():.3f}, d_loss: {d_loss.item():.3f}')

                self.d_optimizer.zero_grad()
                d_loss.backward()  # retain_graph=True)
                # Gradient cliping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.d_optimizer.step()

                d_loss_history.append(d_loss.data)
                
                # Initialize contrastive_loss_value to avoid UnboundLocalError 
                contrastive_loss_value = torch.tensor(0.0).to(device)

                #---- Train cLSTM ----#
                if batch_i > self.config.discriminator_slow_start:
                    if self.config.verbose:
                        tqdm.write('Training cLSTM...')
                    # [seq_len, 1, hidden_size]
                    original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)

                    scores, h_mu, h_log_variance, generated_features = self.summarizer(
                        original_features)
                    _, _, _, uniform_features = self.summarizer(
                        original_features, uniform=True)

                    h_origin, original_prob = self.discriminator(original_features)
                    h_fake, fake_prob = self.discriminator(generated_features)
                    h_uniform, uniform_prob = self.discriminator(uniform_features)
                    tqdm.write(
                        f'original_p: {original_prob.item():.3f}, fake_p: {fake_prob.item():.3f}, uniform_p: {uniform_prob.item():.3f}')

                    # Maximization
                    c_loss = -1 * self.gan_loss(original_prob, fake_prob, uniform_prob)

                    # Contrastive loss 
                    pos_pairs = (h_origin, h_fake)  # Positive pairs
                    neg_pairs = (h_fake, h_uniform)  # Negative pairs (fake vs. uniform)
                    contrastive_loss_value = self.contrastive_loss(pos_pairs, neg_pairs, adaptive_weighting=True)


                    c_loss += contrastive_loss_value

                    tqdm.write(f'gan loss: {gan_loss.item():.3f}, contrastive loss: {contrastive_loss_value.item():.3f}, c_loss: {c_loss.item():.3f}')

                    self.c_optimizer.zero_grad()
                    c_loss.backward()
                    # Gradient cliping
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.config.clip)
                    self.c_optimizer.step()

                    c_loss_history.append(c_loss.data)

                if self.config.verbose:
                    tqdm.write('Plotting...')

                self.writer.update_loss(reconstruction_loss.data, step, 'recon_loss')
                self.writer.update_loss(prior_loss.data, step, 'prior_loss')
                self.writer.update_loss(sparsity_loss.data, step, 'sparsity_loss')
                self.writer.update_loss(gan_loss.data, step, 'gan_loss')
                self.writer.update_loss(contrastive_loss_value.data, step, 'contrastive_loss')
                self.writer.update_loss(original_prob.data, step, 'original_prob')
                self.writer.update_loss(fake_prob.data, step, 'fake_prob')
                self.writer.update_loss(uniform_prob.data, step, 'uniform_prob')

                step += 1

            s_e_loss = torch.stack(s_e_loss_history).mean()
            d_loss = torch.stack(d_loss_history).mean()
            c_loss = torch.stack(c_loss_history).mean()

            # Plot
            
            if self.config.verbose:
                tqdm.write('Plotting...')
            self.writer.update_loss(s_e_loss, epoch_i, 's_e_loss_epoch')
            self.writer.update_loss(d_loss, epoch_i, 'd_loss_epoch')
            self.writer.update_loss(c_loss, epoch_i, 'c_loss_epoch')

            # Save parameters at checkpoint for last epoch
            if (self.n_epoch_trained + epoch_i)%5 == 4:
                checkpoint_path = str(self.config.save_dir) + f'/epoch-{self.n_epoch_trained + epoch_i}.pkl'
                if not os.path.isdir(self.config.save_dir):
                    os.makedirs(self.config.save_dir)
                    
                if self.config.verbose:
                    tqdm.write(f'Save parameters at {checkpoint_path}')
                torch.save({
                            'n_epoch_trained': self.n_epoch_trained + epoch_i + 1,
                            'model_state_dict': self.model.state_dict(),
                            'e_optimizer_state_dict': self.s_e_optimizer.state_dict(),
                            'd_optimizer_state_dict':self.d_optimizer.state_dict(),
                            'c_optimizer_state_dict': self.c_optimizer.state_dict(),
                            }, checkpoint_path)

            self.evaluate(self.n_epoch_trained + epoch_i)

            self.model.train()

    def evaluate(self, epoch_i):

        self.model.eval()

        out_dict = {}

        for video_tensor, video_name in tqdm(self.test_loader, desc='Evaluate', ncols=80, leave=False):

            # [seq_len, batch=1, 2048]
            video_tensor = video_tensor.view(-1, self.config.input_size)
            video_feature = Variable(video_tensor, volatile=True).to(device=device)

            # [seq_len, 1, hidden_size]
            video_feature = self.linear_compress(video_feature.detach()).unsqueeze(1)

            # [seq_len]
            with torch.no_grad():
                scores = self.summarizer.attn(video_feature)
                #print('SCORES', scores)
                scores = scores[0]
                scores = scores.squeeze(1)
                scores = scores.cpu().numpy().tolist()
                
                #print('SCORES', scores)
                out_dict[video_name] = scores

            out_dict[video_name] = scores

            score_save_path = self.config.score_dir.joinpath(f'{self.config.video_type}_{epoch_i}.json')
            if not os.path.isdir(self.config.score_dir):
                os.makedirs(self.config.score_dir)
            with open(score_save_path, 'w') as f:
                tqdm.write(f'Saving score at {str(score_save_path)}.')
                json.dump(out_dict, f)
            score_save_path.chmod(0o777)

    def pretrain(self):
        pass


if __name__ == '__main__':
    pass