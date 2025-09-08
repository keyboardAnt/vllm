# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal V1 example: print + prune logits to an engine-wide hot token set.

This script registers an AdapterLogitsProcessor that, on every decode step:
  - Prints a short trace showing the step length.
  - Retains logits only for a list of "hot" token IDs and sets all
    other ("cold") token logits to -inf, so they cannot be sampled.

Notes:
  - Hot token IDs are read once at adapter init from the environment variable
    FRSPEC_HOT_TOKEN_IDS, which must point to a file path containing a torch
    Tensor (e.g., .pt) or a list of token IDs. If the env var is missing,
    empty, or invalid, an exception is raised.
"""

from vllm import LLM, SamplingParams
from typing import Optional
import os
import torch
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)

def _parse_hot_token_ids_from_env() -> list[int]:
    path = os.environ.get("FRSPEC_HOT_TOKEN_IDS")
    if not path:
        raise ValueError(
            "FRSPEC_HOT_TOKEN_IDS is not set. Set it to a .pt file or list path"
        )
    try:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            ids = obj.to(torch.long).view(-1).tolist()
        elif isinstance(obj, (list, tuple)):
            ids = [int(x) for x in obj]
        elif hasattr(obj, "tolist"):
            ids = [int(x) for x in obj.tolist()]
        else:
            raise ValueError(
                "FRSPEC_HOT_TOKEN_IDS did not load to a tensor/list of ints"
            )
        if not ids:
            raise ValueError(
                "FRSPEC_HOT_TOKEN_IDS file is empty or produced no IDs"
            )
        return ids
    except Exception as e:
        raise ValueError(
            f"Failed to load hot token ids from {path}: {e}"
        )


class FrspecAdapter(AdapterLogitsProcessor):
    """Adapter that prints and prunes logits to a fixed hot token set."""

    def __init__(self, vllm_config, device, is_pin_memory):
        super().__init__(vllm_config, device, is_pin_memory)
        # Engine-wide configuration: read once
        self.hot_token_ids: list[int] = _parse_hot_token_ids_from_env()

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self, params: SamplingParams
    ) -> Optional[RequestLogitsProcessor]:
        # Return a per-request callable: (output_ids, logits) -> logits.
        # Implementation: set all "cold" token logits (not in self.hot_token_ids)
        # to -inf; retain only hot token logits.
        hot_ids = self.hot_token_ids

        def per_req(output_ids, logits):
            print(
                f"[frspec] step_len={len(output_ids)} pruning to hot_token_ids (n={len(hot_ids)})"
            )
            # Filter hot ids to valid range [0, vocab)
            vocab_size = logits.shape[0]
            hot = torch.tensor(hot_ids, dtype=torch.long, device=logits.device)
            hot = hot[(hot >= 0) & (hot < vocab_size)]
            if hot.numel() == 0:
                return logits

            # Retain logits of hot tokens, -inf for the rest
            values_to_keep = logits[hot].clone()
            logits[:] = float("-inf")
            logits[hot] = values_to_keep
            return logits

        return per_req


def main():
    print("[front] before LLM()")
    llm = LLM(model="facebook/opt-125m", logits_processors=[FrspecAdapter])
    print("[front] after LLM()")

    prompt = "The best open source inference engine is "
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=100,
    )

    print("[front] before generate()")
    outputs = llm.generate([prompt], sampling_params)
    print("[front] after generate()")

    # Use the same source as the adapter for verification
    effective_hot_ids = _parse_hot_token_ids_from_env()
    print("[verify] effective hot token ids:", effective_hot_ids)
    for out in outputs:
        seq_out = out.outputs[0]
        print("Generated:", seq_out.text)
        gen_ids = seq_out.token_ids
        print("[verify] generated token count=", len(gen_ids))
        cold = [tid for tid in gen_ids if tid not in effective_hot_ids]
        if cold:
            print("[verify] tokens outside hot set (showing up to 20):", cold[:20])
            raise AssertionError("Generated tokens outside configured hot token set")
        else:
            print("[verify] All generated tokens are within configured hot token set")


if __name__ == "__main__":
    main()
