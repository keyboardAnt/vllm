# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal V1 example: print + prune logits to a fixed hot token set.

This script registers an AdapterLogitsProcessor that, on every decode step:
  - Prints a short trace showing the step length.
  - Retains logits only for a fixed list of "hot" token IDs and sets all
    other ("cold") token logits to -inf, so they cannot be sampled.

Notes:
  - For simplicity, the hot token IDs are a fixed constant below. Wiring an
    external list at init time can be added later.
"""

from vllm import LLM, SamplingParams
from typing import Optional
import torch
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)


# Fixed hot token set for the pruning demo (retain only these tokens).
# All tokens not in this list are assigned -inf and thus never sampled.
HOT_TOKEN_IDS: list[int] = [100, 101, 102, 103, 104]


class FrspecAdapter(AdapterLogitsProcessor):
    """Adapter that prints and prunes logits to a fixed hot token set."""

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self, params: SamplingParams
    ) -> Optional[RequestLogitsProcessor]:
        # Return a per-request callable: (output_ids, logits) -> logits.
        # Implementation: set all "cold" token logits (not in HOT_TOKEN_IDS)
        # to -inf; retain only hot token logits.
        hot_ids = HOT_TOKEN_IDS

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

    for out in outputs:
        seq_out = out.outputs[0]
        print("Generated:", seq_out.text)
        gen_ids = seq_out.token_ids
        print("[verify] generated token count=", len(gen_ids))
        cold = [tid for tid in gen_ids if tid not in HOT_TOKEN_IDS]
        if cold:
            print("[verify] tokens outside HOT_TOKEN_IDS (showing up to 20):", cold[:20])
            raise AssertionError("Generated tokens outside HOT_TOKEN_IDS")
        else:
            print("[verify] All generated tokens are within HOT_TOKEN_IDS")


if __name__ == "__main__":
    main()
