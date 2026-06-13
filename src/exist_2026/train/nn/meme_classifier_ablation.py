import torch
from peft import LoraConfig, get_peft_model, TaskType
from torch import nn
from transformers import AutoModel, CLIPVisionModel

from exist_2026.consts.task_config import Task22, Task21, Task23


class LoRAMemeMultitaskModel(nn.Module):
    """
    Multitask model sharing a single fused representation across subtasks.

    Ablation flags (all default True, preserving original behavior):
        use_film:        if False, skip FiLM modulation; modulated_cls = fused_cls
        use_description: if False, skip description encoding and the cross-attention description-query path
        use_image:       if False, skip image encoding and the cross-attention image-key/value path
        When both use_description and use_image are False, fused_cls falls back
        to the text [CLS], yielding a text-only model.
    """

    def __init__(
            self,
            text_model: str,
            image_model: str,
            lora_r: int = 16,
            lora_alpha: int = 32,
            tasks: set[str] | None = None,
            use_film: bool = True,
            use_description: bool = True,
            use_image: bool = True,
    ):
        super().__init__()
        self.tasks = tasks or {"2.1", "2.2", "2.3"}
        self.use_film = use_film
        self.use_description = use_description
        self.use_image = use_image

        text_encoder = AutoModel.from_pretrained(text_model)
        image_encoder = CLIPVisionModel.from_pretrained(image_model)

        text_lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, r=lora_r, lora_alpha=lora_alpha,
            lora_dropout=0.1, target_modules=["query", "key", "value", "dense"]
        )

        vision_lora_config = LoraConfig(
            task_type=None, r=lora_r, lora_alpha=lora_alpha,
            lora_dropout=0.1, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
        )

        self.text_encoder = get_peft_model(text_encoder, text_lora_config)
        self.image_encoder = get_peft_model(image_encoder, vision_lora_config)

        text_dim = self.text_encoder.config.hidden_size
        image_dim = self.image_encoder.config.hidden_size

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.LayerNorm(256),
            nn.Dropout(0.3)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.LayerNorm(256),
            nn.Dropout(0.3)
        )

        from exist_2026.train.nn.meme_classifier import DemographicEncoder
        self.demo_encoder = DemographicEncoder()

        self.sensorial_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU()
        )

        self.film_gamma = nn.Linear(56, 256)
        self.film_beta = nn.Linear(56, 256)

        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        self.attn_norm = nn.LayerNorm(256)

        self.contrastive_proj = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh()
        )

        if "2.1" in self.tasks:
            self.aux_sexism_head = nn.Sequential(
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Linear(64, 2)
            )

            self.classifier_2_1 = nn.Sequential(
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.4),
                nn.Linear(256, Task21.NUM_CLASSES)
            )

        if "2.2" in self.tasks:
            self.classifier_2_2 = nn.Sequential(
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.4),
                nn.Linear(256, Task22.NUM_CLASSES)
            )

        if "2.3" in self.tasks:
            self.classifier_2_3 = nn.Sequential(
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.4),
                nn.Linear(256, Task23.NUM_CLASSES)
            )

    def _encode_and_fuse(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        t_out = self.text_encoder(
            input_ids=batch["text_input_ids"], attention_mask=batch["text_attention_mask"]
        )
        t_seq = self.text_proj(t_out.last_hidden_state)
        t_cls = t_seq[:, 0, :]

        d_seq = None
        if self.use_description:
            d_out = self.text_encoder(
                input_ids=batch["desc_input_ids"], attention_mask=batch["desc_attention_mask"]
            )
            d_seq = self.text_proj(d_out.last_hidden_state)

        v_seq = None
        if self.use_image:
            v_out = self.image_encoder(pixel_values=batch["pixel_values"])
            v_seq = self.image_proj(v_out.last_hidden_state)

        if self.use_description and self.use_image:
            attn_out, _ = self.cross_attn(query=d_seq, key=v_seq, value=v_seq)
            fused_seq = self.attn_norm(attn_out + d_seq)
            fused_cls = fused_seq[:, 0, :]
        elif self.use_description:
            fused_cls = self.attn_norm(d_seq[:, 0, :])
        elif self.use_image:
            fused_cls = self.attn_norm(v_seq[:, 0, :])
        else:
            fused_cls = t_cls

        if self.use_film:
            demo_feat = self.demo_encoder(
                batch["genders"], batch["ages"], batch["studies"], batch["eths"]
            )
            sens_feat = self.sensorial_encoder(batch["sensorial"])
            human_feat = torch.cat([demo_feat, sens_feat], dim=-1)
            gamma = self.film_gamma(human_feat)
            beta = self.film_beta(human_feat)
            modulated_cls = (fused_cls * (1 + gamma)) + beta
        else:
            modulated_cls = fused_cls

        final_state = torch.cat([t_cls, modulated_cls], dim=-1)
        return final_state, fused_cls

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        final_state, fused_cls = self._encode_and_fuse(batch)

        outputs = {
            "contrast_feat": self.contrastive_proj(final_state)
        }

        if "2.1" in self.tasks:
            outputs["logits_2_1"] = self.classifier_2_1(final_state)
            outputs["log_probs_2_1"] = torch.log_softmax(outputs["logits_2_1"], dim=-1)
            outputs["aux_sexism"] = self.aux_sexism_head(fused_cls)

        if "2.2" in self.tasks:
            outputs["logits_2_2"] = self.classifier_2_2(final_state)
            outputs["log_probs_2_2"] = torch.log_softmax(outputs["logits_2_2"], dim=-1)

        if "2.3" in self.tasks:
            outputs["logits_2_3"] = self.classifier_2_3(final_state)

        return outputs