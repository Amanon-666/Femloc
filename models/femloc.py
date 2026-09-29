"""实现Table II层宽：私有Encoder/Decoder/Mapper与共享Meta-model。"""
import torch
from torch import nn


def network(widths):
    layers = []
    for index, (left, right) in enumerate(zip(widths, widths[1:])):
        layer = nn.Linear(left, right)
        final = index == len(widths) - 2
        if final:
            nn.init.xavier_uniform_(layer.weight)
        else:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        nn.init.zeros_(layer.bias)
        layers.append(layer)
        if not final:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def private(m, config, seed):
    torch.manual_seed(seed)
    c = config["model"]
    return (network([m, *c["encoder_hidden"], c["latent"]]).to(config["device"]),
            network([c["latent"], *c["decoder_hidden"], m]).to(config["device"]),
            network([c["representation"], *c["mapper_hidden"], c["output"]]).to(config["device"]))


def shared(config, seed):
    torch.manual_seed(seed)
    c = config["model"]
    return network([c["latent"], *c["shared_hidden"], c["representation"]]).to(config["device"])


class Localizer(nn.Module):
    def __init__(self, encoder, shared_model, mapper):
        super().__init__()
        self.encoder, self.shared, self.mapper = encoder, shared_model, mapper

    def forward(self, x):
        return self.mapper(self.shared(self.encoder(x)))
