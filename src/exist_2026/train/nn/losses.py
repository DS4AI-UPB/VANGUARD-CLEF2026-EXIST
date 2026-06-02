import torch
import torch.nn.functional as fun
from torch import nn


def supervised_contrastive_loss(
        embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """Pulls embeddings of samples with the same label together and pushes apart embeddings with different labels."""
    embeddings = fun.normalize(embeddings, p=2, dim=1)
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    device = embeddings.device
    labels = labels.contiguous().view(-1, 1)
    positive_mask = torch.eq(labels, labels.T).float().to(device)
    logits_mask = torch.scatter(
        torch.ones_like(positive_mask),
        1,
        torch.arange(positive_mask.shape[0]).view(-1, 1).to(device),
        0
    )
    positive_mask = positive_mask * logits_mask

    if positive_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    exp_logits = torch.exp(similarity_matrix) * logits_mask
    log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
    mean_log_prob_pos = (positive_mask * log_prob).sum(1) / (positive_mask.sum(1) + 1e-6)
    return -mean_log_prob_pos.mean()


def kl_soft_loss(log_probs: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    return fun.kl_div(log_probs, target_dist, reduction="batchmean", log_target=False)


def multilabel_soft_bce_loss(logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    return fun.binary_cross_entropy_with_logits(logits, target_dist, reduction="mean")


class MultitaskLoss(torch.nn.Module):
    """
    Weighted combination of per-task losses.
    """

    def __init__(
            self,
            weight_2_1: float = 1.0,
            weight_2_2: float = 1.0,
            weight_2_3: float = 1.0,
            weight_aux: float = 0.3,
            weight_contrastive: float = 0.1,
            tasks: set[str] | None = None,
    ):
        """
        :param weight_2_1: Weight for subtask 2.1 KL loss.
        :param weight_2_2: Weight for subtask 2.2 KL loss.
        :param weight_2_3: Weight for subtask 2.3 BCE loss.
        :param weight_aux: Weight for auxiliary sexism head (on fused_cls).
        :param weight_contrastive: Weight for supervised contrastive loss.
        :param tasks: Which subtasks are active.
        """
        super().__init__()
        self.w21 = weight_2_1
        self.w22 = weight_2_2
        self.w23 = weight_2_3
        self.w_aux = weight_aux
        self.w_con = weight_contrastive
        self.tasks = tasks or {"2.1", "2.2", "2.3"}

    def forward(
            self,
            model_outputs: dict[str, torch.Tensor],
            batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute total loss and return individual loss components for logging.
        """
        device = next(iter(model_outputs.values())).device
        total = torch.tensor(0.0, device=device)
        log = {}

        if "2.1" in self.tasks and "log_probs_2_1" in model_outputs:
            loss_21 = kl_soft_loss(model_outputs["log_probs_2_1"], batch["target_2_1"])
            total = total + self.w21 * loss_21
            log["loss_2_1"] = loss_21.item()

            # Auxiliary sexism head
            if "aux_sexism" in model_outputs:
                hard_labels = batch["target_2_1"].argmax(dim=1)
                aux_loss = fun.cross_entropy(model_outputs["aux_sexism"], hard_labels)
                total = total + self.w_aux * aux_loss
                log["loss_aux"] = aux_loss.item()

        if "2.2" in self.tasks and "log_probs_2_2" in model_outputs:
            loss_22 = kl_soft_loss(model_outputs["log_probs_2_2"], batch["target_2_2"])
            total = total + self.w22 * loss_22
            log["loss_2_2"] = loss_22.item()

        if "2.3" in self.tasks and "logits_2_3" in model_outputs:
            loss_23 = multilabel_soft_bce_loss(model_outputs["logits_2_3"], batch["target_2_3"])
            total = total + self.w23 * loss_23
            log["loss_2_3"] = loss_23.item()

        if self.w_con > 0 and "contrast_feat" in model_outputs and "target_2_1" in batch:
            hard_labels = batch["target_2_1"].argmax(dim=1)
            con_loss = supervised_contrastive_loss(model_outputs["contrast_feat"], hard_labels)
            total = total + self.w_con * con_loss
            log["loss_contrastive"] = con_loss.item()

        log["loss_total"] = total.item()
        return total, log


def compute_loss(outputs: dict, batch: dict, aux_weight: float = 0.3, contrast_weight=0.1) -> torch.Tensor:
    """
    Multi-objective loss:
        l = kl_uncertainty + aux_w * aux_BCE + contrastive_w * contrastive

    KL divergence is weighted by exp(-(p*(1-p))) to focus on high-consensus samples (p near 0 or 1).
    """
    p_yes = batch["target_2_1"][:, 1]

    uncertainty_weight = torch.exp(-(p_yes * (1 - p_yes)))
    kl_loss = (
            nn.KLDivLoss(reduction="none")(outputs["log_probs"], batch["target_2_1"]).sum(dim=1) * uncertainty_weight
    ).mean()

    aux_loss = nn.BCEWithLogitsLoss()(outputs["aux_sexism"][:, 1], p_yes)

    binary_labels = (p_yes >= 0.5).long()
    contrast_loss = supervised_contrastive_loss(outputs["contrast_feat"], binary_labels)

    return kl_loss + (aux_weight * aux_loss) + (contrast_weight * contrast_loss)
