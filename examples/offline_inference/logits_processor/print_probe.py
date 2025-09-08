# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal script to validate that a logits processor is being called (V1).

Implements an AdapterLogitsProcessor that prints each time it is invoked for a
request during decoding. Register it at LLM construction time (V1-compliant).
"""

from vllm import LLM, SamplingParams
from typing import Optional
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)


class PrintProbeAdapter(AdapterLogitsProcessor):
    """Adapter that prints when per-request logits are processed (no-op)."""

    def is_argmax_invariant(self) -> bool:
        # Does not change selection; purely observational.
        # NOTE: Set to False to force invocation under greedy decoding
        # (temperature=0.0); V1 may skip argmax-invariant processors.
        return False

    def new_req_logits_processor(
        self, params: SamplingParams
    ) -> Optional[RequestLogitsProcessor]:
        # Return a per-request callable: (output_ids, logits) -> logits
        def per_req(output_ids, logits):
            print(
                f"[print_probe] called: step_len={len(output_ids)}, logits_shape={tuple(logits.shape)}"
            )
            return logits

        return per_req


def main():
    print("[front] before LLM()")
    llm = LLM(model="facebook/opt-125m", logits_processors=[PrintProbeAdapter])
    print("[front] after LLM()")

    prompt = "The best open source inference engine is "
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=5,
    )

    print("[front] before generate()")
    outputs = llm.generate([prompt], sampling_params)
    print("[front] after generate()")

    for out in outputs:
        print("Generated:", out.outputs[0].text)


if __name__ == "__main__":
    main()
