"""ATLAS model: a LoRA-tuned causal LM with one learned control embedding per source."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import Linear as LoraLinear



A2_B_INIT_STD = 1e-4


class _LoraRoute:
    """Passes each row's source index to the per-source LoRA layers."""

    __slots__ = ("index", "uniform_k")

    def __init__(self):
        self.index = None
        self.uniform_k = 0

    def set(self, index, uniform_k):
        self.index, self.uniform_k = index, uniform_k


class PerSourceLoraLinear(LoraLinear):
    """peft's LoRA Linear with lora_B replaced by a [K, d_out, r] parameter, indexed per row."""

    @classmethod
    def install(cls, layer, num_sources, route, init_std, generator=None):
        adapters = list(layer.lora_A.keys())
        assert len(adapters) == 1, f"per-source B expects exactly one adapter, got {adapters}"
        name = adapters[0]
        out_f, r = layer.lora_B[name].weight.shape
        layer.__class__ = cls
        layer._route = route
        layer._adapter_name = name
        w = torch.randn(num_sources, out_f, r, generator=generator,
                        dtype=layer.lora_B[name].weight.dtype,
                        device=layer.lora_B[name].weight.device) * init_std
        layer.lora_B_k = nn.Parameter(w)
        layer._freeze_shadow_b()
        return layer

    def materialized_B(self):
        """The [K, d_out, r] up-projection stack the routing consumes."""
        return self.lora_B_k

    def _freeze_shadow_b(self):
        with torch.no_grad():
            self.lora_B[self._adapter_name].weight.zero_()
        self.lora_B[self._adapter_name].weight.requires_grad_(False)

    def set_adapter(self, adapter_names, inference_mode: bool = False) -> None:
        """peft's BaseTunerLayer.set_adapter sets requires_grad=True on every module named in"""
        super().set_adapter(adapter_names, inference_mode=inference_mode)
        self._freeze_shadow_b()

    def forward(self, x, *args, **kwargs):
        self._check_forward_args(x, *args, **kwargs)
        kwargs.pop("adapter_names", None)
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)
        assert not self.merged, "per-source B does not support a merged adapter"
        result = self.base_layer(x, *args, **kwargs)
        out_dtype = result.dtype
        name = self._adapter_name
        lora_A, dropout = self.lora_A[name], self.lora_dropout[name]
        x = self._cast_input_dtype(x, lora_A.weight.dtype)
        h = lora_A(dropout(x))                                  # [N, L, r]
        k = self._route.uniform_k
        B = self.materialized_B()                               # [K, d_out, r]
        if k is not None:
            delta = F.linear(h, B[k])
        else:
            idx = self._route.index
            assert idx is not None, ("per-source LoRA B: no route set (see ATLASModel._route_rows).")
            assert idx.numel() == h.shape[0], (
                f"route length {idx.numel()} != batch {h.shape[0]}; the row->source index is stale")
            Bn = B.index_select(0, idx)                          # [N, d_out, r]
            delta = torch.bmm(h, Bn.transpose(1, 2))            # [N, L, d_out]
        return (result + delta * self.scaling[name]).to(out_dtype)


def _masked_mean(x, mask):
    """[B, L] x [B, L] → [B]"""
    return (x * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)


def doc_score(theta_lp, mask):
    """The document score: mean_t log p_theta(x_t | x_<t, c_k), length-normalised. [B,K,S] -> [B,K]"""
    return _masked_mean(theta_lp, mask.unsqueeze(1))


def _prod_readout(theta_lp, loss_mask, T):
    """Per-token pairwise product readout over the K conditionings. [B, K, L-1] x [B, L-1] -> [B, K]

    For each token, exp(-sum_j relu(s_j - s_k) / T): 1 when source k has the highest log-prob,
    decaying with how far it falls behind the others. Averaged over the valid tokens."""
    m = (loss_mask > 0.5)
    lp = theta_lp.float()
    d = lp.unsqueeze(2) - lp.unsqueeze(1)                       # [B,K,K,L]  d[b,k,j,t]=s_k-s_j
    v = torch.exp(-(torch.relu(-d) / float(T)).sum(dim=2))
    v = v * m.unsqueeze(1)
    return v.sum(dim=2) / m.sum(dim=1, keepdim=True).clamp(min=1)


class ATLASModel(nn.Module):
    def __init__(
        self,
        base_model_path: str,
        source_names: list[str],
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj"),
        turn_lora: bool = True,
        data_type=torch.bfloat16,
        tail_chunks: int = 4,
        prefix_len: int = 1,
        per_source_lora_b: bool = False,
    ):
        super().__init__()
        assert prefix_len >= 0
        self.prefix_len = int(prefix_len)
        self.source_names = list(source_names)
        self.num_sources = len(self.source_names)
        self.turn_lora = turn_lora
        self.tail_chunks = tail_chunks

        theta_path = base_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(theta_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        _attn = os.environ.get("ATLAS_ATTN", "sdpa")
        _attn_kw = {"attn_implementation": _attn} if _attn else {}
        theta_lm = AutoModelForCausalLM.from_pretrained(theta_path, torch_dtype=data_type, **_attn_kw)
        theta_lm.config.use_cache = False

        if turn_lora:
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(target_modules),
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(theta_lm, lora_cfg)
        else:
            self.model = theta_lm

        self.per_source_lora_b = bool(per_source_lora_b)
        self._route = _LoraRoute()
        self._per_source_layers = []
        if self.per_source_lora_b:
            if not turn_lora:
                raise ValueError("per_source_lora_b requires turn_lora=True")
            for name, mod in self.model.named_modules():
                if isinstance(mod, LoraLinear):
                    PerSourceLoraLinear.install(mod, self.num_sources, self._route,
                                                A2_B_INIT_STD)
                    self._per_source_layers.append((name, mod))
            if not self._per_source_layers:
                raise RuntimeError("per_source_lora_b: no LoRA Linear layers found to convert")

        hidden_size = theta_lm.config.hidden_size
        emb_table = self.model.get_input_embeddings().weight
        emb_mean = emb_table.mean(dim=0).detach().to(torch.float32)
        emb_std = float(emb_table.detach().float().std())

        init = (emb_mean.view(1, 1, -1).repeat(self.num_sources, self.prefix_len, 1)
                + 0.01 * emb_std * torch.randn(self.num_sources, self.prefix_len, hidden_size))
        if self.prefix_len == 0:
            self.register_buffer("control_embeddings", init.to(data_type), persistent=False)
        else:
            self.control_embeddings = nn.Parameter(init.to(data_type))   # [K, P, H]

    @property
    def device(self):
        return next(self.model.parameters()).device

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _route_rows(self, index, uniform_k=None):
        """Install the row -> source index every per-source LoRA layer reads."""
        self._route.set(index, uniform_k)

    def _backbone(self, *, input_ids=None, inputs_embeds=None, attention_mask=None):
        """Run the decoder without the LM head. Returns hidden states [B, L, H]."""
        return self.model.get_decoder()(
            input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
        ).last_hidden_state

    def _lm_head(self, hidden):
        """Apply the frozen LM head."""
        return self.model.get_output_embeddings()(hidden)

    def _theta_pass(self, input_ids, attention_mask):
        """theta log-probs [B, K, L-1] under every control code."""
        B, L = input_ids.shape
        targets = input_ids[:, 1:].unsqueeze(-1)
        tok_emb = self.model.get_input_embeddings()(input_ids)              # [B, L, H]

        P = self.prefix_len
        exp_tok = tok_emb.unsqueeze(0).expand(self.num_sources, B, L, -1).reshape(self.num_sources * B, L, -1)
        e_full = self.control_embeddings.unsqueeze(1).expand(self.num_sources, B, P, -1)  # [K, B, P, H]
        exp_ctrl = e_full.reshape(self.num_sources * B, P, e_full.shape[-1])
        inputs_emb = torch.cat([exp_ctrl, exp_tok], dim=1)                  # [K*B, L+P, H]
        src_rows = (torch.arange(self.num_sources, device=input_ids.device)
                    .view(-1, 1).expand(self.num_sources, B).reshape(-1))   # [K*B]
        self._route_rows(src_rows, uniform_k=None)

        ctrl_mask = attention_mask.new_ones(self.num_sources * B, P)
        exp_attn = attention_mask.unsqueeze(0).expand(self.num_sources, B, L).reshape(self.num_sources * B, L)
        full_attn = torch.cat([ctrl_mask, exp_attn], dim=1)

        hidden = self._backbone(inputs_embeds=inputs_emb, attention_mask=full_attn)  # [K*B, L+P, H]

        def tail(hid_slice, tgt_slice):                                     # [B, S, H] -> [B, S]
            log_s = F.log_softmax(self._lm_head(hid_slice).float(), dim=-1)
            return log_s.gather(-1, tgt_slice).squeeze(-1)

        theta_lp_list = []
        for k in range(self.num_sources):
            hid = hidden[k * B:(k + 1) * B, P:-1, :]                        # [B, L-1, H]
            n = max(1, min(self.tail_chunks, hid.size(1)))
            theta_lp_list.append(torch.cat(
                [tail(h_c, t_c) for h_c, t_c in zip(hid.chunk(n, dim=1), targets.chunk(n, dim=1))],
                dim=1))                                                     # [B, L-1]
        del hidden
        return torch.stack(theta_lp_list, dim=1)                            # [B, K, L-1]

    @torch.no_grad()
    def get_score_vectors(self, input_ids, attention_mask, prod_temps=()):
        """Both readouts from one forward pass. Shape [B, K, 1 + len(prod_temps)]:
        column 0 is the mean log-prob score (argmax), then one prod readout per temperature."""
        loss_mask = attention_mask[:, 1:].float()
        theta_lp = self._theta_pass(input_ids, attention_mask)
        cols = [doc_score(theta_lp, loss_mask)]
        cols += [_prod_readout(theta_lp, loss_mask, t) for t in prod_temps]
        return torch.stack(cols, dim=-1)

    @torch.no_grad()
    def _bk_diagnostics(self):
        """Cosine-similarity and norm statistics of the per-source LoRA up-projections, for logging."""
        if not self._per_source_layers:
            return {}
        K = self.num_sources
        iu = torch.triu_indices(K, K, offset=1)
        acc, mx, nrm, n = 0.0, -1.0, 0.0, 0
        for _, lay in self._per_source_layers:
            b = lay.materialized_B().detach().float().reshape(K, -1)
            bn = F.normalize(b, dim=1)
            pairs = (bn @ bn.T)[iu[0], iu[1]]
            acc += float(pairs.mean()); mx = max(mx, float(pairs.max()))
            nrm += float(b.norm(dim=1).mean()); n += 1
        dev = self.control_embeddings.device
        out = {"Bk_cos_mean": torch.tensor(acc / n, device=dev),
               "Bk_cos_max": torch.tensor(mx, device=dev),
               "Bk_norm_mean": torch.tensor(nrm / n, device=dev)}
        return out

    def _theta_pass_single(self, input_ids, attention_mask, labels):
        """Per-token log p_theta(x_t | x_<t, c_y) under each row's own source code. Shape [B, L-1]."""
        B, L = input_ids.shape
        P = self.prefix_len
        tok_emb = self.model.get_input_embeddings()(input_ids)               # [B, L, H]
        ctrl = self.control_embeddings.index_select(0, labels)              # [B, P, H]
        inputs_emb = torch.cat([ctrl, tok_emb], dim=1)                      # [B, L+P, H]
        full_attn = torch.cat([attention_mask.new_ones(B, P), attention_mask], dim=1)
        self._route_rows(labels, uniform_k=None)
        hidden = self._backbone(inputs_embeds=inputs_emb, attention_mask=full_attn)
        hid = hidden[:, P:-1, :]                                            # [B, L-1, H]
        targets = input_ids[:, 1:].unsqueeze(-1)
        del hidden
        use_ckpt = self.training and torch.is_grad_enabled()

        def tail(h_c, t_c):
            log_s = F.log_softmax(self._lm_head(h_c).float(), dim=-1)
            return log_s.gather(-1, t_c).squeeze(-1)                        # [B, S]

        n = max(1, min(self.tail_chunks, hid.size(1)))
        parts = [checkpoint(tail, h_c, t_c, use_reentrant=False) if use_ckpt else tail(h_c, t_c)
                 for h_c, t_c in zip(hid.chunk(n, dim=1), targets.chunk(n, dim=1))]
        return torch.cat(parts, dim=1)                                      # [B, L-1]

    def compute_losses(self, input_ids, attention_mask, labels, loss_mask=None, lam_sft=1.0):
        """SFT loss: next-token NLL of each document under its own source's control code."""
        if bool((labels >= self.num_sources).any()):
            raise ValueError("batch contains human rows (label K) but the model has no human column; "
                             "train with --human_class or drop human rows")
        if loss_mask is None:
            loss_mask = attention_mask[:, 1:].float()
        else:
            loss_mask = loss_mask[:, 1:].float()
        true_lp = self._theta_pass_single(input_ids, attention_mask, labels)
        L_sft = -_masked_mean(true_lp, loss_mask).mean()
        out = {"loss": lam_sft * L_sft, "L_sft": L_sft}
        out.update(self._bk_diagnostics())
        return out

    def forward(self, input_ids, attention_mask, labels, loss_mask=None, lam_sft=1.0):
        return self.compute_losses(input_ids, attention_mask, labels, loss_mask=loss_mask, lam_sft=lam_sft)

    def save_weights(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        if self.turn_lora:
            self.model.save_pretrained(out_dir)
        else:
            self.model.save_pretrained(os.path.join(out_dir, "theta_full"))
        extra = {
            "control_embeddings": self.control_embeddings.detach().cpu(),
            "source_names": self.source_names,
            "turn_lora": self.turn_lora,
            "prefix_len": self.prefix_len,
            "per_source_lora_b": self.per_source_lora_b,
        }
        if self.per_source_lora_b:
            extra["lora_B_k"] = {n: m.lora_B_k.detach().cpu()
                                 for n, m in self._per_source_layers}
        torch.save(extra, os.path.join(out_dir, "atlas_extra.pt"))

    def load_weights(self, ckpt_dir: str, trainable: bool = False):
        """Load a checkpoint. Pass trainable=True to keep the adapter trainable, e.g. to resume training."""
        extra_path = os.path.join(ckpt_dir, "atlas_extra.pt")
        if not os.path.exists(extra_path):
            extra_path = os.path.join(ckpt_dir, "trace_extra.pt")
        extra = torch.load(extra_path, map_location="cpu")
        assert extra.get("turn_lora", True) == self.turn_lora, (
            f"checkpoint turn_lora={extra.get('turn_lora')} but model turn_lora={self.turn_lora}"
        )
        if self.turn_lora:
            self.model.load_adapter(ckpt_dir, adapter_name="default", is_trainable=trainable,
                                    torch_device="cpu")
        else:
            loaded = AutoModelForCausalLM.from_pretrained(
                os.path.join(ckpt_dir, "theta_full"), torch_dtype=self.control_embeddings.dtype,
            )
            self.model.load_state_dict(loaded.state_dict())
            del loaded
        saved = extra["control_embeddings"]
        if saved.dim() == 2:
            saved = saved.unsqueeze(1)
        if saved.shape != self.control_embeddings.shape:
            assert saved.shape[0] == self.num_sources and saved.shape[2] == self.control_embeddings.shape[2], (
                f"checkpoint control_embeddings {tuple(saved.shape)} is not a prefix-length "
                f"variant of {tuple(self.control_embeddings.shape)}")
            self.control_embeddings = nn.Parameter(
                saved.to(self.control_embeddings.dtype).to(self.control_embeddings.device))
            self.prefix_len = int(saved.shape[1])
            print(f"[load_weights] prefix_len -> {self.prefix_len} (from checkpoint)")
        else:
            with torch.no_grad():
                self.control_embeddings.copy_(saved.to(self.control_embeddings.dtype))


        ckpt_psl = bool(extra.get("per_source_lora_b", "lora_B_k" in extra))
        if ckpt_psl and not self.per_source_lora_b:
            if not self.turn_lora:
                raise ValueError("checkpoint has per-source lora_B_k but turn_lora=False")
            for name, mod in self.model.named_modules():
                if isinstance(mod, LoraLinear) and not isinstance(mod, PerSourceLoraLinear):
                    PerSourceLoraLinear.install(mod, self.num_sources, self._route,
                                                A2_B_INIT_STD)
                    self._per_source_layers.append((name, mod))
            self.per_source_lora_b = True
            print(f"[load_weights] per_source_lora_b -> True "
                  f"({len(self._per_source_layers)} layers, from checkpoint)")
        elif self.per_source_lora_b and not ckpt_psl:
            raise ValueError(
                f"per_source_lora_b=True but {ckpt_dir} is a shared-B checkpoint; load it without "
                "--per_source_lora_b.")

        if self.per_source_lora_b:
            if "lora_B_k" not in extra:
                raise ValueError(
                    f"{extra_path} claims per_source_lora_b but carries no "
                    "'lora_B_k'.")
            saved_b = extra["lora_B_k"]
            with torch.no_grad():
                for n, m in self._per_source_layers:
                    assert n in saved_b, f"lora_B_k missing layer {n}"
                    assert saved_b[n].shape == m.lora_B_k.shape, (
                        f"lora_B_k[{n}] {tuple(saved_b[n].shape)} != {tuple(m.lora_B_k.shape)}")
                    m.lora_B_k.copy_(saved_b[n].to(m.lora_B_k.dtype))
            for _, m in self._per_source_layers:
                m._freeze_shadow_b()

        return self
