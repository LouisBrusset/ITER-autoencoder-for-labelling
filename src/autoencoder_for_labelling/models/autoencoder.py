import torch
import torch.nn as nn
from typing import List, Optional


class Convolutional_Beta_VAE(nn.Module):
    """A simple 1D Convolutional Variational AutoEncoder with configurable
    number of conv layers and filter size. This class accepts inputs of shape
    (batch, input_dim) where input_dim is treated as the temporal length and
    a single channel is assumed. Internally the tensor is reshaped to
    (batch, 1, input_dim) and reconstructed back to (batch, input_dim).

    Parameters:
    - input_dim: int length of the 1D input
    - encoding_dim: int latent size
    - conv_layers: int number of conv layers (N). If 0, falls back to a small
      fully-connected encoder/decoder around the latent.
    - conv_filter_size: int kernel size for all conv layers (same for all)
    - encoder_layer_sizes/decoder_layer_sizes: Optional list of dense sizes
      used after conv flattening / before conv unflattening.
    """
    def __init__(self, input_dim: int = 20, encoding_dim: int = 8,
                 conv_layers: int = 0, conv_filter_size: int = 3,
                 encoder_layer_sizes: Optional[List[int]] = None,
                 decoder_layer_sizes: Optional[List[int]] = None):
        super().__init__()
        self.input_dim = int(input_dim)
        self.encoding_dim = int(encoding_dim)
        self.conv_layers = int(conv_layers)
        self.conv_filter_size = int(conv_filter_size)

        encoder_layer_sizes = encoder_layer_sizes or []
        decoder_layer_sizes = decoder_layer_sizes or []

        if self.conv_layers <= 0:
            # Fallback to simple dense VAE-like encoder/decoder
            enc = []
            prev = self.input_dim
            for sz in encoder_layer_sizes:
                enc.append(nn.Linear(prev, int(sz)))
                enc.append(nn.ReLU())
                prev = int(sz)
            # produce mu and logvar
            self.fc_mu = nn.Linear(prev, self.encoding_dim)
            self.fc_logvar = nn.Linear(prev, self.encoding_dim)
            self.encoder_fc = nn.Sequential(*enc) if enc else nn.Identity()

            # decoder fc
            dec = []
            prev = self.encoding_dim
            for sz in decoder_layer_sizes:
                dec.append(nn.Linear(prev, int(sz)))
                dec.append(nn.ReLU())
                prev = int(sz)
            dec.append(nn.Linear(prev, self.input_dim))
            dec.append(nn.Sigmoid())
            self.decoder_fc = nn.Sequential(*dec)
            self._use_conv = False
        else:
            # Build conv encoder
            convs = []
            in_ch = 1
            # Build conv encoder with stride=2 (downsampling)
            lengths = []
            convs = []
            in_ch = 1
            cur_length = self.input_dim
            out_chs = []
            for i in range(self.conv_layers):
                out_ch = max(4, 8 * (2 ** i))
                pad = (self.conv_filter_size - 1) // 2
                # stride=2 for downsampling
                convs.append(nn.Conv1d(in_ch, out_ch, kernel_size=self.conv_filter_size, padding=pad, stride=2))
                convs.append(nn.ReLU())
                # compute output length after this conv
                cur_length = (cur_length + 2 * pad - (self.conv_filter_size - 1) - 1) // 2 + 1
                lengths.append(cur_length)
                out_chs.append(out_ch)
                in_ch = out_ch
            self.conv_encoder = nn.Sequential(*convs)
            # record conv metadata for decoder reconstruction
            self._conv_lengths = lengths
            self._conv_out_channels = in_ch
            self._conv_out_length = cur_length
            self._conv_out_chs = out_chs
            self._conv_flatten_dim = self._conv_out_channels * self._conv_out_length

            # optional dense encoder layers
            enc_dense = []
            prev = self._conv_flatten_dim
            for sz in encoder_layer_sizes:
                enc_dense.append(nn.Linear(prev, int(sz)))
                enc_dense.append(nn.ReLU())
                prev = int(sz)
            self.encoder_dense = nn.Sequential(*enc_dense) if enc_dense else nn.Identity()

            # final mu/logvar from prev
            self.fc_mu = nn.Linear(prev, self.encoding_dim)
            self.fc_logvar = nn.Linear(prev, self.encoding_dim)

            # Decoder: map latent back to conv feature map
            # start with linear that outputs (conv_out_channels * conv_out_length)
            dec_dense = []
            prev = self.encoding_dim
            for sz in decoder_layer_sizes:
                dec_dense.append(nn.Linear(prev, int(sz)))
                dec_dense.append(nn.ReLU())
                prev = int(sz)
            dec_dense.append(nn.Linear(prev, self._conv_flatten_dim))
            self.decoder_dense = nn.Sequential(*dec_dense)

            # conv transpose stack (mirror of encoder) with stride=2 (upsampling)
            convt = []
            out_chs = self._conv_out_chs
            # reverse for transpose
            prev_ch = out_chs[-1]
            # current length starts as final conv out length
            current_length = self._conv_out_length
            for i in range(self.conv_layers - 1, -1, -1):
                out_ch = out_chs[i-1] if i - 1 >= 0 else 1
                pad = (self.conv_filter_size - 1) // 2
                # compute desired target length after this transpose (previous encoder length or input_dim)
                target_length = self._conv_lengths[i - 1] if i - 1 >= 0 else self.input_dim
                # ConvTranspose1d output length formula: (L_in -1)*stride - 2*pad + kernel + output_padding
                # solve for output_padding
                kernel = self.conv_filter_size
                stride = 2
                out_len_no_op = (current_length - 1) * stride - 2 * pad + kernel
                output_padding = target_length - out_len_no_op
                # ensure output_padding is within valid range [0, stride-1]
                if output_padding < 0 or output_padding >= stride:
                    # clamp to 0 to avoid errors (best-effort)
                    output_padding = 0
                convt.append(nn.ConvTranspose1d(prev_ch, out_ch, kernel_size=self.conv_filter_size, padding=pad, stride=2, output_padding=output_padding))
                # for last layer, apply sigmoid, otherwise relu
                if i == 0:
                    convt.append(nn.Sigmoid())
                else:
                    convt.append(nn.ReLU())
                # after this layer the resulting length should be target_length
                current_length = target_length
                prev_ch = out_ch

            self.conv_decoder = nn.Sequential(*convt)
            self._use_conv = True

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        # x shape: (batch, input_dim)
        if not self._use_conv:
            h = self.encoder_fc(x)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            z = self.reparameterize(mu, logvar)
            recon = self.decoder_fc(z)
            return recon, mu, logvar
        else:
            b = x.shape[0]
            x_conv = x.view(b, 1, self.input_dim)
            h = self.conv_encoder(x_conv)
            h_flat = h.view(b, -1)
            h2 = self.encoder_dense(h_flat)
            mu = self.fc_mu(h2)
            logvar = self.fc_logvar(h2)
            z = self.reparameterize(mu, logvar)
            dec = self.decoder_dense(z)
            # reshape according to conv decoder expected channels and length
            dec = dec.view(b, self._conv_out_channels, self._conv_out_length)
            recon = self.conv_decoder(dec)
            # recon shape (b,1,L) -> (b,L)
            recon = recon.view(b, self.input_dim)
            return recon, mu, logvar

    def encode(self, x: torch.Tensor):
        out = self.forward(x)
        if isinstance(out, tuple):
            _, mu, logvar = out
            return mu, logvar
        return out

    def decode(self, z: torch.Tensor):
        # produce reconstruction from latent z
        if not self._use_conv:
            return self.decoder_fc(z)
        dec = self.decoder_dense(z)
        b = z.shape[0]
        dec = dec.view(b, self._conv_out_channels, self._conv_out_length)
        recon = self.conv_decoder(dec)
        return recon.view(b, self.input_dim)

