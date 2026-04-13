# Instruction Scheduling and Pipeline Guide

## CDNA3 Pipeline Model

Each CU has 4 SIMD units. Each SIMD:

- Executes one wavefront instruction per cycle
- Round-robins among ready wavefronts (TLP hides latency)

Latency hiding: If wavefront A stalls on memory, SIMD executes wavefront B, C, D, etc.

## Instruction-Level Parallelism (ILP)

When occupancy is low (few wavefronts), ILP within a single wavefront becomes critical.

**Goal**: Keep the pipeline saturated by interleaving independent instructions.

```
// Bad: dependency chain -> pipeline stall
global_load v0, ...
s_waitcnt vmcnt(0)    // stalls here
v_add_f32 v1, v0, v2  // must wait for load

// Good: interleave independent work
global_load v0, ...    // issue load
v_mul_f32 v3, v4, v5   // independent VALU work
s_add_u32 s0, s0, 1    // independent SALU work (can dual-issue!)
s_waitcnt vmcnt(0)     // load likely completed by now
v_add_f32 v1, v0, v2   // use the loaded data
```

## s_waitcnt Strategy

**Principle**: Wait as late as possible, wait as little as possible.

| Pattern | Code | Reason |
|---------|------|--------|
| Immediate wait | `s_waitcnt vmcnt(0)` right after load | Bad: kills ILP |
| Deferred wait | Load, do other work, then wait | Good: hides latency |
| Partial wait | `vmcnt(N)` where N = allowed outstanding operations remaining | Best: minimal stall |

## Dual-Issue Rules (CDNA3)

VALU + SALU can issue in the same cycle, provided:

1. No register dependency between them
2. VALU uses VGPR, SALU uses SGPR
3. Both are ready (no pending waits)

**Optimization**: Pair address calculations (SALU) with data operations (VALU).

## MFMA Scheduling

MFMA instructions have high latency (64 cycles) but can overlap with other work:

```
// Pipeline: while MFMA N executes, load data for N+1
v_mfma_f32_32x32x8_bf16 a[0:15], v[0:3], v[4:7], a[0:15]  // 64 cycles
global_load_dwordx4 v[0:3], ...  // issue during MFMA latency
global_load_dwordx4 v[4:7], ...  // issue during MFMA latency
s_waitcnt vmcnt(0)               // loads should be complete by now
v_mfma_f32_32x32x8_bf16 a[0:15], v[0:3], v[4:7], a[0:15]  // next MFMA
```

---

## MFMA Dependency Resolution Rules (CDNA4, ISA Table 38)

The following is an excerpt from the **CDNA4 ISA documentation dependency table (Table 38)** relevant to agent scheduling in this repository: used to determine how many **NOPs** (or equivalent independent work) must be inserted between two instructions to avoid RAW/WAR hazards. Values depend on the specific **MFMA variant**; without forwarding, plan ILP for the maximum interval.

| Scenario | NOP Requirement |
|----------|-----------------|
| Non-MFMA VALU writes to a VGPR -> MFMA reads the **same** VGPR | **2** (need 2 NOPs) |
| MFMA writes -> **same** MFMA reads as **SrcC** (accumulation, **exactly same** opcode) | **0** (forwarding supported) |
| MFMA writes -> **different** MFMA reads as SrcA/B | **5 / 8 / 12 / 20** (depends on MFMA variant, **no** forwarding) |
| MFMA writes -> VALU / VM / LDS / FLAT reads **overlapping destination** registers | **5 / 8 / 12 / 20** |
| SGEMM writes -> SGEMM reads SrcC (same destination) | **0** (forwarding) |
| `V_CMPX` writes **EXEC** -> MFMA | **4** (no exec mask forwarding) |
| XDL / SMFMAC reads SrcC -> VALU writes (WAR) | **1 / 3 / 7 / 15** (depends on variant) |

**Implications for the agent**:

1. **Back-to-back MFMA accumulation** (same opcode, SrcC chaining): Can leverage **0 NOP** forwarding, but other hazards (e.g., LDS/VMEM) must still be satisfied.
2. **Switching SrcA/B sources** or **interleaving MFMA with VALU/LDS**: Must insert sufficient **independent instructions** (or `s_nop`) per the table above -- do not assume the same behavior as SrcC.
3. **MFMA adjacent to `V_CMPX` modifying EXEC**: Reserve **4 NOP**-level spacing.
4. When designing **8-wave ping-pong**, **sched_group_barrier** patterns, apply the above rules to verify whether the inner unrolling is sufficient to "fill" the pipeline.

> The complete matrix is in the AMD CDNA4 ISA PDF; different `v_mfma_*` / `v_smfmac_*` shapes correspond to different NOP columns.
