"""
GPU Compute Tutorial: High-Performance Tiled Matrix Multiplication
==================================================================

This example teaches GPU *performance engineering* using matrix multiplication
as the running example.  It builds on compute_tiled_matmul.py and adds:

Topics covered
--------------
1.  Choosing workgroup sizes     – occupancy analysis, warp alignment, tile sizing
2.  Memory tiling                – cooperative tile loading into workgroup memory
3.  Avoiding bank conflicts      – padding the shared-memory stride by 1
4.  Minimising global memory     – data-reuse analysis: reads reduced by TILE×
5.  Occupancy considerations     – workgroup-memory budget vs concurrency

The running example is  C = A @ B  where  A ∈ ℝ^(M×K)  and  B ∈ ℝ^(K×N).

Run this file directly::

    python examples/compute_perf_matmul.py
"""

import struct
import math
import numpy as np
import wgpu

# ===========================================================================
# PERFORMANCE KNOB 1 – TILE SIZE
# ===========================================================================
# TILE is the side length of the square tile loaded into workgroup memory each
# iteration.  The workgroup dimensions are (TILE, TILE, 1), so there are
# TILE*TILE threads per workgroup.
#
# Effect on performance:
#   Larger TILE → more data reuse per global-memory fetch (good) but:
#     • more workgroup memory per workgroup → fewer concurrent workgroups
#       on each SM (lower occupancy, potentially bad)
#     • more registers per thread (can also lower occupancy)
#
# Typical sweet spots: 8, 16, 32
#   TILE=8  : small workgroup (64 threads), low memory pressure, easy to run
#             many workgroups concurrently.
#   TILE=16 : 256 threads per workgroup, usually best trade-off on modern GPUs.
#   TILE=32 : 1024 threads, maximum work per workgroup but high memory use.
#
# The occupancy formula (simplified):
#   max_workgroups_per_SM = min(
#       floor(max_threads_per_SM / (TILE * TILE)),      # thread limit
#       floor(workgroup_memory_per_SM / shared_mem_used) # memory limit
#   )
#
# For TILE=16 on a typical GPU (max 2048 threads/SM, 48 KB shared/SM):
#   thread limit  : 2048 / 256 = 8 workgroups
#   shared_mem_used per workgroup = 2 tiles × 16×17 floats × 4 bytes = 2176 B
#     (the ×17 accounts for the bank-conflict padding added below)
#   memory limit  : 49152 / 2176 ≈ 22 workgroups  → thread-bound, 8 active
# ===========================================================================
TILE = 16

# ===========================================================================
# PERFORMANCE KNOB 2 – BANK-CONFLICT PADDING
# ===========================================================================
# Workgroup (shared) memory is divided into 32 banks (on most hardware), each
# 4 bytes wide.  A float at byte offset b resides in bank (b / 4) % 32.
#
# Bank conflict: two or more threads in the same warp access *different*
# addresses that map to the *same* bank.  This forces serial accesses, costing
# one extra memory cycle per conflicting thread.
#
# The problem in a square tile of width TILE stored as a 1-D array
# tile[lrow * TILE + lcol]:
#   - When TILE = 16, all threads in a warp that share the same lcol value
#     access addresses spaced TILE * 4 = 64 bytes apart.
#   - 64 / 4 = 16.  Each thread lands on bank (base + 16*t) % 32, cycling
#     through banks {0,16,0,16,...} for t = 0,1,2,3,...  → 2-way conflict.
#   - When TILE = 32: spacing = 128 bytes → 128/4 = 32 ≡ 0 (mod 32).  All
#     threads in the same warp column hit the *same* bank → 32-way conflict!
#
# Fix: store each row with stride TILE_PAD = TILE + 1.
#   Address of tile[row][col] = tile[row * TILE_PAD + col]
#   Row spacing = TILE_PAD * 4 = (TILE+1) * 4 bytes.
#   For TILE=16: (17 * 4) = 68 bytes → bank (68/4) % 32 = 17 % 32 = 17.
#     Each successive row starts in a different bank → no two-way conflict.
#   For TILE=32: (33 * 4) = 132 bytes → bank 33 % 32 = 1 ≠ 0 → conflict-free.
#
# Cost: one extra float per row of padding wastes (TILE * 4) bytes of workgroup
#       memory per tile – a small overhead, usually worth the throughput gain.
# ===========================================================================
TILE_PAD = TILE + 1  # padded row stride to eliminate bank conflicts

# Matrix dimensions (edit to experiment with larger matrices)
M = 128   # rows of A / rows of C
K = 96    # cols of A / rows of B
N = 64    # cols of B / cols of C

rng = np.random.default_rng(0)
A = rng.random((M, K), dtype=np.float32)
B = rng.random((K, N), dtype=np.float32)

# ===========================================================================
# WGSL SHADER – BANK-CONFLICT-FREE TILED MATRIX MULTIPLICATION
# ===========================================================================
# Key differences vs compute_tiled_matmul.py:
#   1. tile_a and tile_b use TILE_PAD (= TILE+1) as the row stride, not TILE.
#      This eliminates bank conflicts when threads in the same warp column all
#      read from the same tile row (column-access pattern during accumulation).
#   2. Inline comments explain every performance decision.
#
# Global-memory access count analysis:
#   Naive matmul: each thread reads K floats from A and K floats from B
#     → 2*M*N*K global reads total.
#   Tiled matmul: each tile of A (TILE×TILE elements) is read once by the
#     workgroup → (M/TILE)*(N/TILE) workgroups × (K/TILE) tiles × TILE²
#     reads = M*N*K/TILE global reads for A (and same for B).
#     Total = 2*M*N*K/TILE  →  TILE× reduction in global memory traffic.
# ===========================================================================
shader_source = f"""
// ── Storage buffers ──────────────────────────────────────────────────────────
@group(0) @binding(0) var<storage, read>       mat_a : array<f32>;
@group(0) @binding(1) var<storage, read>       mat_b : array<f32>;
@group(0) @binding(2) var<storage, read_write> mat_c : array<f32>;
@group(0) @binding(3) var<uniform>             dims  : vec4<u32>;
// dims.x = M, dims.y = K, dims.z = N, dims.w = padding (unused)

// ── Workgroup (shared) memory with bank-conflict padding ──────────────────────
//
// TILE_PAD = TILE + 1 is the padded row stride.
//
// Layout of tile_a (row-major, padded):
//   tile_a[lrow * {TILE_PAD}u + lcol]  →  element (lrow, lcol) of the A-tile
//
// Without padding (stride = TILE):
//   Two threads with lid.x == 0 but different lid.y values would access
//   addresses spaced TILE * 4 bytes apart.  When TILE is a multiple of the
//   number of banks (32), they land on the same bank → bank conflict.
//
// With padding (stride = TILE + 1):
//   The spacing is (TILE+1) * 4 bytes, which is not a multiple of 128, so
//   successive rows fall in different banks → conflict-free access.
var<workgroup> tile_a: array<f32, {TILE * TILE_PAD}>;
var<workgroup> tile_b: array<f32, {TILE * TILE_PAD}>;

// ── Compute entry point ───────────────────────────────────────────────────────
@compute @workgroup_size({TILE}, {TILE}, 1)
fn main(
    @builtin(global_invocation_id) gid : vec3<u32>,
    @builtin(local_invocation_id)  lid : vec3<u32>,
) {{
    let row  = gid.y;    // global row in C (and A)
    let col  = gid.x;    // global col in C (and B)
    let lrow = lid.y;    // local row within the tile (0 .. TILE-1)
    let lcol = lid.x;    // local col within the tile (0 .. TILE-1)

    let M = dims.x;
    let K = dims.y;
    let N = dims.z;

    var acc: f32 = 0.0;

    let num_tiles = (K + {TILE}u - 1u) / {TILE}u;

    for (var t: u32 = 0u; t < num_tiles; t++) {{

        // ── Load tile of A ────────────────────────────────────────────────────
        // Thread (lrow, lcol) loads A[row, t*TILE + lcol].
        //
        // Coalescing: all threads in the same warp (same lrow, consecutive lcol)
        // read consecutive addresses of mat_a → one wide memory transaction.
        let a_col = t * {TILE}u + lcol;
        tile_a[lrow * {TILE_PAD}u + lcol] = select(
            0.0,
            mat_a[row * K + a_col],
            row < M && a_col < K,
        );

        // ── Load tile of B ────────────────────────────────────────────────────
        // Thread (lrow, lcol) loads B[t*TILE + lrow, col].
        //
        // Coalescing: all threads in the same warp (same lrow, consecutive lcol)
        // read consecutive addresses of mat_b → one wide memory transaction.
        let b_row = t * {TILE}u + lrow;
        tile_b[lrow * {TILE_PAD}u + lcol] = select(
            0.0,
            mat_b[b_row * N + col],
            b_row < K && col < N,
        );

        // ── Barrier: wait for all threads to finish loading ───────────────────
        // No thread may read tile_a or tile_b until every thread has written
        // its element.  Without this barrier the computation is a data race.
        workgroupBarrier();

        // ── Accumulate the partial dot-product ────────────────────────────────
        // Each thread (lrow, lcol) computes:
        //   acc += sum_k tile_a[lrow, k] * tile_b[k, lcol]
        //
        // tile_a access pattern: fixed lrow, varying k → row access, sequential
        //   bank indices (k % 32) → no conflict.
        // tile_b access pattern: varying k (= row index), fixed lcol → column
        //   access.  With stride TILE_PAD = TILE+1, bank(k) = (k*(TILE+1)) % 32,
        //   which cycles through all 32 banks without repetition when TILE+1 is
        //   not divisible by 32 → conflict-free.
        for (var k: u32 = 0u; k < {TILE}u; k++) {{
            acc += tile_a[lrow * {TILE_PAD}u + k]
                 * tile_b[k    * {TILE_PAD}u + lcol];
        }}

        // ── Barrier: wait before overwriting tiles on next iteration ──────────
        workgroupBarrier();
    }}

    if (row < M && col < N) {{
        mat_c[row * N + col] = acc;
    }}
}}
"""

# ===========================================================================
# DEVICE / BUFFER SETUP
# ===========================================================================
adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
device = adapter.request_device_sync()

buf_a = device.create_buffer_with_data(
    data=A.flatten().tobytes(),
    usage=wgpu.BufferUsage.STORAGE,
)
buf_b = device.create_buffer_with_data(
    data=B.flatten().tobytes(),
    usage=wgpu.BufferUsage.STORAGE,
)
buf_c = device.create_buffer(
    size=M * N * 4,
    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
)
buf_dims = device.create_buffer_with_data(
    data=struct.pack("IIII", M, K, N, 0),
    usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
)

# ===========================================================================
# PIPELINE SETUP
# ===========================================================================
binding_layouts = [
    {
        "binding": 0,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
    },
    {
        "binding": 1,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
    },
    {
        "binding": 2,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {"type": wgpu.BufferBindingType.storage},
    },
    {
        "binding": 3,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {"type": wgpu.BufferBindingType.uniform},
    },
]
bindings = [
    {"binding": 0, "resource": {"buffer": buf_a, "offset": 0, "size": buf_a.size}},
    {"binding": 1, "resource": {"buffer": buf_b, "offset": 0, "size": buf_b.size}},
    {"binding": 2, "resource": {"buffer": buf_c, "offset": 0, "size": buf_c.size}},
    {"binding": 3, "resource": {"buffer": buf_dims, "offset": 0, "size": buf_dims.size}},
]

bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
pipeline_layout = device.create_pipeline_layout(
    bind_group_layouts=[bind_group_layout]
)
bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

cshader = device.create_shader_module(code=shader_source)
compute_pipeline = device.create_compute_pipeline(
    layout=pipeline_layout,
    compute={"module": cshader, "entry_point": "main"},
)

# ===========================================================================
# DISPATCH
# ===========================================================================
# Dispatch a 2-D grid: one workgroup covers a TILE×TILE block of the output.
wg_x = math.ceil(N / TILE)
wg_y = math.ceil(M / TILE)

command_encoder = device.create_command_encoder()
compute_pass = command_encoder.begin_compute_pass()
compute_pass.set_pipeline(compute_pipeline)
compute_pass.set_bind_group(0, bind_group)
compute_pass.dispatch_workgroups(wg_x, wg_y, 1)
compute_pass.end()
device.queue.submit([command_encoder.finish()])

# ===========================================================================
# VERIFY
# ===========================================================================
raw = device.queue.read_buffer(buf_c)
C_gpu = np.frombuffer(raw, dtype=np.float32).reshape(M, N)
C_ref = A @ B

max_err = float(np.max(np.abs(C_gpu - C_ref)))
assert np.allclose(C_gpu, C_ref, atol=1e-4), f"Mismatch! Max error: {max_err}"
print(
    f"Bank-conflict-free tiled matmul "
    f"({M}×{K}) @ ({K}×{N}) = ({M}×{N}):  OK  (max_err={max_err:.2e})"
)
print(f"  TILE={TILE}, TILE_PAD={TILE_PAD}, workgroups dispatched: {wg_x}×{wg_y}")

# ===========================================================================
# PERFORMANCE SUMMARY (printed for learning)
# ===========================================================================
threads_per_wg = TILE * TILE
shared_mem_bytes = 2 * TILE * TILE_PAD * 4  # two padded tiles, 4 bytes/float
global_reads_naive = 2 * M * N * K
global_reads_tiled = 2 * M * N * K // TILE
print("\n── Performance summary ────────────────────────────────────────────────")
print(f"  Threads per workgroup      : {threads_per_wg}")
print(f"  Shared memory per workgroup: {shared_mem_bytes} bytes ({shared_mem_bytes / 1024:.2f} KB)")
print(f"  Global float reads (naive) : {global_reads_naive:,}")
print(f"  Global float reads (tiled) : {global_reads_tiled:,}  ({TILE}× reduction)")
print(
    f"  Occupancy estimate (TILE={TILE}): "
    f"thread-bound ≈ {2048 // threads_per_wg} workgroups/SM "
    f"(assuming 2048 max threads/SM)"
)
print("───────────────────────────────────────────────────────────────────────")

# ===========================================================================
# EXERCISES
# ===========================================================================
# 1. Change TILE to 8 and 32.  The printed summary shows how shared-memory
#    usage and estimated occupancy change.  For large matrices, use
#    compute_timestamps.py to measure actual throughput for each TILE value.
#
# 2. Set TILE_PAD = TILE (remove the padding) and re-run.  The results are
#    still correct, but on real hardware you may observe lower performance due
#    to bank conflicts.  Use a GPU profiler (Nsight, RenderDoc) to confirm.
#
# 3. The shader uses `select(false_val, true_val, cond)` for branch-free
#    boundary clamping.  Replace it with an explicit `if` block and compare
#    performance.  On GPU, divergent branches inside a warp serialize execution.
#
# 4. Implement a version where each thread computes a 2×2 block of output
#    elements (register tiling).  This further reduces the ratio of memory
#    accesses to FMAs and is used in production GEMM libraries.
#
# 5. Add GPU timestamp queries (see compute_timestamps.py) to measure actual
#    kernel execution time and compute GFLOP/s.
#    FLOP count for matmul: 2 * M * N * K  (one multiply + one add per element
#    of the inner product).

