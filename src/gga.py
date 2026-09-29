"""GGA early weight search, adapted from official ERM_GGA for coordinate loss."""

from itertools import combinations

import torch
from torch.nn import functional as F


def regression_loss(model, batch: dict) -> torch.Tensor:
    return (model(batch) - batch["xy"]).square().sum(dim=1).mean()


def merge_batches(batches: list[dict]) -> dict:
    return {key: torch.cat([batch[key] for batch in batches], dim=0) for key in batches[0]}


def minimum_gradient_cosine(model, batches: list[dict]) -> float:
    parameters = tuple(model.parameters())
    gradients = []
    for batch in batches:
        grads = torch.autograd.grad(regression_loss(model, batch), parameters)
        gradients.append(torch.cat([part.flatten() for part in grads]))
    return min(F.cosine_similarity(gradients[i], gradients[j], dim=0).item()
               for i, j in combinations(range(len(gradients)), 2))


def gga_step(model, optimizer, batches: list[dict], cfg: dict) -> tuple[float, float, float, int]:
    """Keep the official cumulative perturbation/acceptance rule, then take one ERM update."""
    pooled = merge_batches(batches)
    best_similarity = minimum_gradient_cosine(model, batches)
    best_loss = regression_loss(model, pooled).item()
    initial_similarity = best_similarity
    base = [parameter.detach().clone() for parameter in model.parameters()]
    best = [parameter.detach().clone() for parameter in model.parameters()]
    accepted = 0

    for _ in range(cfg["gga_search_steps"]):
        with torch.no_grad():
            for parameter in model.featurizer.parameters():
                parameter.add_((torch.rand_like(parameter) * 2 - 1) * cfg["gga_neighborhood_size"])

        candidate_similarity = minimum_gradient_cosine(model, batches)
        candidate_loss = regression_loss(model, pooled).item()
        # Coordinate scaling is arbitrary, so apply the original gate to relative loss.
        if candidate_loss < best_loss * (1 + cfg["gga_relative_loss_tolerance"]) and candidate_similarity > best_similarity:
            best_similarity, best_loss = candidate_similarity, candidate_loss
            best = [parameter.detach().clone() for parameter in model.parameters()]
            accepted += 1
            # The source implementation resets to the original step weights only after acceptance.
            with torch.no_grad():
                for parameter, original in zip(model.parameters(), base):
                    parameter.copy_(original)

    with torch.no_grad():
        for parameter, winner in zip(model.parameters(), best):
            parameter.copy_(winner)

    optimizer.zero_grad()
    loss = regression_loss(model, pooled)
    loss.backward()
    optimizer.step()
    return loss.item(), initial_similarity, best_similarity, accepted


def erm_step(model, optimizer, batches: list[dict]) -> float:
    optimizer.zero_grad()
    loss = regression_loss(model, merge_batches(batches))
    loss.backward()
    optimizer.step()
    return loss.item()
