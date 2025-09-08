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
        return True

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
    llm = LLM(model="facebook/opt-125m", logits_processors=[PrintProbeAdapter])

    prompt = "Hello, my name is"
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=5,
    )

    outputs = llm.generate([prompt], sampling_params)
    for out in outputs:
        print("Generated:", out.outputs[0].text)


if __name__ == "__main__":
    main()
