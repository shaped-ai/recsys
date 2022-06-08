import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(pl.LightningModule):
    def __init__(
        self,
        data_schema,
        num_hidden=2,
        hidden_dim=600,
        latent_dim=200,
        dropout=0.5,
        learning_rate=1e-3,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.latent_dim = latent_dim

        # Input dropout
        self.input_dropout = nn.Dropout(p=dropout)
        num_items = data_schema["n_items"]

        # Construct a list of dimensions for the encoder and the decoder
        dims = [hidden_dim] * 2 * num_hidden
        dims = [num_items] + dims + [latent_dim * 2]

        # Stack encoders and decoders
        encoder_modules, decoder_modules = [], []
        for i in range(len(dims) // 2):
            encoder_modules.append(nn.Linear(dims[2 * i], dims[2 * i + 1]))
            if i == 0:
                decoder_modules.append(nn.Linear(dims[-1] // 2, dims[-2]))
            else:
                decoder_modules.append(nn.Linear(dims[-2 * i - 1], dims[-2 * i - 2]))

        self.encoder = nn.ModuleList(encoder_modules)
        self.decoder = nn.ModuleList(decoder_modules)

        # Initialize weights
        self.encoder.apply(self.weight_init)
        self.decoder.apply(self.weight_init)

        # TODO beta stuff
        self.beta = 0.0

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.zero_()

    def forward(self, x, training=True):
        x = F.normalize(x)
        x = self.input_dropout(x)

        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i != len(self.encoder) - 1:
                x = torch.tanh(x)

        mu, logvar = x[:, : self.latent_dim], x[:, self.latent_dim :]

        if training:
            sigma = torch.exp(0.5 * logvar)
            eps = torch.randn_like(sigma)
            x = mu + eps * sigma
        else:
            x = mu

        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i != len(self.decoder) - 1:
                x = torch.tanh(x)

        return x, mu, logvar

    def training_step(self, batch, batch_idx):
        x = batch
        recon_x, mu, logvar = self(x)
        CE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        return CE + self.beta * KLD

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return [optimizer]
