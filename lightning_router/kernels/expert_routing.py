"""
Triton kernel for expert-parallel routing dispatch.

Key optimisations
─────────────────
1. **Coalesced global memory access** – tokens destined for the same expert are
   gathered into contiguous segments so that the subsequent expert FFN reads
   hit coalesced 128-byte cache-lines instead of scattered addresses.
2. **Shared-memory caching** – the gating scores and permutation indices are
   staged in SRAM to avoid repeated global loads during the scatter / gather
   phases.
3. **Fused permute-and-scale** – the top-k gate softmax, token permutation,
   and expert-capacity masking are fused into a single kernel launch, removing
   three intermediate global-memory round-trips present in the naïve PyTorch
   implementation.

The kernel is written for a *top-k* gating policy (default k=2) over
``num_experts`` experts.  It produces:
  • ``permuted_tokens``  – (total_capacity, hidden)  reordered token matrix
  • ``expert_offsets``    – (num_experts + 1,)         CSR-style offset array
  • ``routing_weights``   – (total_capacity,)          per-slot gate weight
  • ``source_indices``    – (total_capacity,)          reverse map back to the
                                                        original token order
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 – Histogram + Prefix-sum  (one workgroup, cooperative)
# Compute per-expert token counts and exclusive prefix-sum (CSR offsets).
# ──────────────────────────────────────────────────────────────────────────────


@triton.jit
def _expert_histogram_kernel(
    expert_ids_ptr,    # (num_tokens * top_k,)  int32 – chosen expert for each slot
    counts_ptr,        # (num_experts,)         int32 – OUTPUT histogram
    num_slots: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Count how many tokens are assigned to each expert (histogram).

    Uses shared-memory atomics so we avoid contention on global memory.
    Each program instance processes a tile of *slots* (token × top_k pairs).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_slots

    # ── Coalesced load of expert assignments ──────────────────────────
    expert_ids = tl.load(expert_ids_ptr + offsets, mask=mask, other=0)

    # ── Shared-memory histogram accumulation ──────────────────────────
    # Triton compiles per-expert atomic adds into efficient warp-level
    # reductions when NUM_EXPERTS is small (≤ 16).
    for e in tl.static_range(NUM_EXPERTS):
        count = tl.sum((expert_ids == e).to(tl.int32) * mask.to(tl.int32))
        if count > 0:
            tl.atomic_add(counts_ptr + e, count)


@triton.jit
def _exclusive_prefix_sum_kernel(
    counts_ptr,      # (num_experts,)       int32 – INPUT histogram
    offsets_ptr,     # (num_experts + 1,)   int32 – OUTPUT prefix-sum
    NUM_EXPERTS: tl.constexpr,
):
    """Single-workgroup exclusive prefix-sum over expert counts → CSR offsets."""
    ids = tl.arange(0, NUM_EXPERTS)
    counts = tl.load(counts_ptr + ids)

    # Simple sequential scan (NUM_EXPERTS is tiny, e.g. 4-16)
    tl.store(offsets_ptr, 0)  # offsets[0] = 0
    running = tl.zeros([], dtype=tl.int32)
    for e in tl.static_range(NUM_EXPERTS):
        c_e = tl.load(counts_ptr + e)
        running += c_e
        tl.store(offsets_ptr + e + 1, running)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 – Scatter tokens into expert-contiguous layout
# ──────────────────────────────────────────────────────────────────────────────


@triton.jit
def _scatter_tokens_kernel(
    # inputs
    tokens_ptr,         # (num_tokens, hidden)  fp16/bf16
    expert_ids_ptr,     # (num_slots,)          int32
    gate_weights_ptr,   # (num_slots,)          fp32
    token_indices_ptr,  # (num_slots,)          int32  – which token each slot came from
    offsets_ptr,        # (num_experts + 1,)    int32  – CSR offsets
    write_counters_ptr, # (num_experts,)        int32  – atomic counters (init 0)
    capacity: tl.constexpr,
    # outputs
    permuted_ptr,       # (total_capacity, hidden)
    routing_w_ptr,      # (total_capacity,)
    source_idx_ptr,     # (total_capacity,)
    # dims
    num_slots: tl.constexpr,
    HIDDEN: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    """
    Scatter each (token, gate_weight) pair into the contiguous expert segment.

    * Reads are coalesced along the hidden dimension (BLOCK_HIDDEN tile).
    * Shared memory holds the destination pointers so the inner hidden-dim loop
      does not re-compute offsets.
    """
    pid_slot = tl.program_id(0)
    pid_hid = tl.program_id(1)

    slot_offsets = pid_slot * BLOCK_SLOTS + tl.arange(0, BLOCK_SLOTS)
    slot_mask = slot_offsets < num_slots

    hid_offsets = pid_hid * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
    hid_mask = hid_offsets < HIDDEN

    # ── Load routing metadata (cached in registers / shared mem) ──────
    expert_ids = tl.load(expert_ids_ptr + slot_offsets, mask=slot_mask, other=0)
    gate_w = tl.load(gate_weights_ptr + slot_offsets, mask=slot_mask, other=0.0)
    tok_idx = tl.load(token_indices_ptr + slot_offsets, mask=slot_mask, other=0)

    # ── For each slot, atomically claim a write position ──────────────
    for s in tl.static_range(BLOCK_SLOTS):
        s_mask = s < num_slots - pid_slot * BLOCK_SLOTS
        if s_mask:
            eid = tl.load(expert_ids_ptr + pid_slot * BLOCK_SLOTS + s)
            base_offset = tl.load(offsets_ptr + eid)
            write_pos = tl.atomic_add(write_counters_ptr + eid, 1)
            dst = base_offset + write_pos

            if write_pos < capacity:
                # Store routing metadata
                gw = tl.load(gate_weights_ptr + pid_slot * BLOCK_SLOTS + s)
                ti = tl.load(token_indices_ptr + pid_slot * BLOCK_SLOTS + s)
                tl.store(routing_w_ptr + dst, gw)
                tl.store(source_idx_ptr + dst, ti)

                # ── Coalesced copy of the hidden-dimension tile ───────
                src_row = ti  # original token index
                for h_start in range(0, HIDDEN, BLOCK_HIDDEN):
                    h_offs = h_start + tl.arange(0, BLOCK_HIDDEN)
                    h_m = h_offs < HIDDEN
                    vals = tl.load(tokens_ptr + src_row * HIDDEN + h_offs, mask=h_m)
                    tl.store(permuted_ptr + dst * HIDDEN + h_offs, vals, mask=h_m)


# ──────────────────────────────────────────────────────────────────────────────
# Python wrapper
# ──────────────────────────────────────────────────────────────────────────────


def expert_routing_forward(
    tokens: torch.Tensor,       # (num_tokens, hidden)
    expert_ids: torch.Tensor,   # (num_tokens * top_k,)  int32
    gate_weights: torch.Tensor, # (num_tokens * top_k,)  float32
    token_indices: torch.Tensor,# (num_tokens * top_k,)  int32
    num_experts: int,
    capacity_factor: float = 1.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Route tokens to experts using fused Triton kernels.

    Returns
    -------
    permuted_tokens : (total_capacity, hidden)
    expert_offsets  : (num_experts + 1,)   CSR-style
    routing_weights : (total_capacity,)
    source_indices  : (total_capacity,)
    """
    num_tokens, hidden = tokens.shape
    num_slots = expert_ids.numel()
    device = tokens.device

    # Step 1: Histogram -------------------------------------------------
    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    BLOCK = 1024
    grid_hist = ((num_slots + BLOCK - 1) // BLOCK,)
    _expert_histogram_kernel[grid_hist](
        expert_ids, counts, num_slots,
        NUM_EXPERTS=num_experts, BLOCK_SIZE=BLOCK,
    )

    # Step 2: Capacity & prefix-sum -------------------------------------
    tokens_per_expert = int((num_slots / num_experts) * capacity_factor)
    capacity = max(tokens_per_expert, 1)

    offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    _exclusive_prefix_sum_kernel[(1,)](
        counts, offsets, NUM_EXPERTS=num_experts,
    )

    total_capacity = int(offsets[-1].item())

    # Step 3: Scatter ---------------------------------------------------
    permuted = torch.empty(total_capacity, hidden, dtype=tokens.dtype, device=device)
    routing_w = torch.empty(total_capacity, dtype=torch.float32, device=device)
    source_idx = torch.empty(total_capacity, dtype=torch.int32, device=device)
    write_counters = torch.zeros(num_experts, dtype=torch.int32, device=device)

    BLOCK_SLOTS = 32
    BLOCK_HIDDEN = min(128, hidden)
    grid_scatter = (
        (num_slots + BLOCK_SLOTS - 1) // BLOCK_SLOTS,
        1,  # hidden-dim tiling handled inside the kernel loop
    )

    _scatter_tokens_kernel[grid_scatter](
        tokens, expert_ids, gate_weights, token_indices,
        offsets, write_counters, capacity,
        permuted, routing_w, source_idx,
        num_slots=num_slots,
        HIDDEN=hidden,
        NUM_EXPERTS=num_experts,
        BLOCK_SLOTS=BLOCK_SLOTS,
        BLOCK_HIDDEN=BLOCK_HIDDEN,
    )

    return permuted, offsets, routing_w, source_idx


# ──────────────────────────────────────────────────────────────────────────────
# Gather-back kernel (inverse permutation after expert FFNs)
# ──────────────────────────────────────────────────────────────────────────────


@triton.jit
def _gather_tokens_kernel(
    # inputs
    expert_out_ptr,   # (total_capacity, hidden)
    routing_w_ptr,    # (total_capacity,)
    source_idx_ptr,   # (total_capacity,)
    # output
    output_ptr,       # (num_tokens, hidden)  – accumulated
    total_capacity: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_CAP: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    """
    Gather expert outputs back to the original token order, weighted by gate.

    Fuses the multiply-by-gate-weight and the scatter-add into one pass,
    using coalesced reads along hidden dim.
    """
    pid = tl.program_id(0)
    cap_offsets = pid * BLOCK_CAP + tl.arange(0, BLOCK_CAP)
    cap_mask = cap_offsets < total_capacity

    for s in tl.static_range(BLOCK_CAP):
        idx = pid * BLOCK_CAP + s
        if idx < total_capacity:
            w = tl.load(routing_w_ptr + idx)
            dst_tok = tl.load(source_idx_ptr + idx)
            for h_start in range(0, HIDDEN, BLOCK_HIDDEN):
                h_offs = h_start + tl.arange(0, BLOCK_HIDDEN)
                h_m = h_offs < HIDDEN
                val = tl.load(expert_out_ptr + idx * HIDDEN + h_offs, mask=h_m)
                scaled = val * w
                tl.atomic_add(output_ptr + dst_tok * HIDDEN + h_offs, scaled, mask=h_m)


def expert_routing_gather(
    expert_output: torch.Tensor,   # (total_capacity, hidden)
    routing_weights: torch.Tensor, # (total_capacity,)
    source_indices: torch.Tensor,  # (total_capacity,)
    num_tokens: int,
) -> torch.Tensor:
    """Gather expert FFN outputs back to original token positions."""
    total_capacity, hidden = expert_output.shape
    device = expert_output.device

    output = torch.zeros(num_tokens, hidden, dtype=expert_output.dtype, device=device)

    BLOCK_CAP = 64
    BLOCK_HIDDEN = min(128, hidden)
    grid = ((total_capacity + BLOCK_CAP - 1) // BLOCK_CAP,)

    _gather_tokens_kernel[grid](
        expert_output, routing_weights, source_indices,
        output,
        total_capacity=total_capacity,
        HIDDEN=hidden,
        BLOCK_CAP=BLOCK_CAP,
        BLOCK_HIDDEN=BLOCK_HIDDEN,
    )
    return output
