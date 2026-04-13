# FlashInfer on ROCm Reference

FlashInfer is an attention kernel library optimized for LLM serving, with native paged KV-cache support. It is ported to AMD GPUs with MFMA-native kernels.

---

## 1. Supported Hardware and Software

| Hardware | Architecture | GPU IDs |
|----------|-------------|---------|
| MI300X, MI325X | CDNA3 | gfx942 |
| MI355X, MI350X | CDNA4 | gfx950 |

**ROCm versions:** 6.4+, 7.0.2, 7.1.1, 7.2

---

## 2. Feature Matrix

| Kernel Type | FP16/BF16 | FP8 |
|-------------|-----------|-----|
| Decode Attention | Supported | Supported |
| Prefill Attention | Supported | Work in progress |

### Attention Variants

- **MHA** (Multi-Head Attention)
- **GQA** (Grouped Query Attention)
- **MQA** (Multi-Query Attention)
- Paged KV-cache (variable-length sequences, no padding waste)
- Ragged tensors
- `torch.compile` compatible
- AITER backend (experimental)

---

## 3. API Examples

### Single Decode with KV Cache

```python
import flashinfer

kv_len, num_kv_heads, head_dim = 2048, 32, 128
k = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)
v = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)

q = torch.randn(num_kv_heads, head_dim).half().to(0)
o = flashinfer.single_decode_with_kv_cache(q, k, v)
```

### Batch Prefill with Paged KV Cache

```python
import flashinfer

# Create a paged KV cache handler
workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer)

# Plan the operation (call once per batch shape)
prefill_wrapper.plan(
    qo_indptr=qo_indptr,
    kv_indptr=kv_indptr,
    kv_indices=kv_indices,
    kv_last_page_len=kv_last_page_len,
    num_qo_heads=num_qo_heads,
    num_kv_heads=num_kv_heads,
    head_dim=head_dim,
    page_size=page_size,
)

# Run the attention
output = prefill_wrapper.run(q, kv_data)
```

---

## 4. Porting from NVIDIA to AMD: Key Changes

The AMD port required significant changes due to architectural differences:

| Aspect | NVIDIA (Original) | AMD (Ported) |
|--------|-------------------|--------------|
| Warp size | 32 threads | 64 threads (wavefront) |
| Matrix instruction | wmma | MFMA |
| Matrix tile geometry | Varies by wmma | 16x16 tile (MFMA) |
| Shared memory banks | 32 banks | 32 banks (CDNA3), 64 banks (CDNA4) |
| Bank conflict rules | Different | Updated indexing required |

The port modified shared memory access patterns and reindexed data layouts to match MFMA's 16x16 matrix tile geometry.

---

## 5. Performance Context

### Position in the Attention Kernel Hierarchy

For AMD GPUs, the typical performance ranking (best to worst) is:

```
AITER assembly > CK FMHA > Triton FA > FlashInfer
```

FlashInfer's strength is not raw peak performance but rather its **serving-oriented feature set**: paged KV-cache, variable-length sequences, GQA/MQA, and `torch.compile` compatibility.

### When to Use FlashInfer vs Alternatives

| Scenario | Recommendation |
|----------|----------------|
| LLM serving with paged KV-cache | FlashInfer -- first-class paged attention support |
| Maximum attention throughput | AITER (assembly path) or CK FMHA |
| DeepSeek MLA decode | AITER MLA (specialized kernel) |
| Quick prototyping | Triton Flash Attention |
| General-purpose serving (vLLM) | Let vLLM auto-select (AITER > CK > FlashInfer) |

---

## 6. Integration with Frameworks

### vLLM

FlashInfer can serve as an attention backend in vLLM on ROCm. vLLM's backend selection priority is typically AITER > CK > FlashInfer, but FlashInfer may be selected for specific attention patterns (e.g., paged attention with variable-length sequences).

### SGLang

FlashInfer is used as one of the available attention backends in SGLang on ROCm.

---

## 7. Key Limitations

1. **FP8 prefill**: Still work in progress as of April 2026.
2. **Performance gap**: Generally slower than AITER assembly and CK FMHA for raw attention throughput on AMD GPUs.
3. **Architecture coverage**: Only CDNA3 (gfx942) and CDNA4 (gfx950); no RDNA support.

---

## Related Documentation

- AITER attention kernels (higher performance alternative): `libraries/aiter-ops-reference.md`
- CK-Tile Flash Attention implementation: `libraries/ck-tile-tuning.md`
- GEMM tuning (related compute paths): `libraries/gemm-tuning-guide.md`
