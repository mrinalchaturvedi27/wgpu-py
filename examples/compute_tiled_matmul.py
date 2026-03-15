"""
GPU Compute Tutorial: Tiled Matrix Multiplication with Workgroup Memory
=======================================================================

This example teaches advanced GPU compute patterns by implementing *tiled*
matrix multiplication – the canonical workgroup-shared-memory benchmark.

Topics covered
--------------
1.  Naive vs tiled matrix multiplication (why tiling helps)
2.  Workgroup (shared) memory  – fast on-chip SRAM shared within a workgroup
3.  Memory barriers  – workgroupBarrier() to synchronise threads
4.  2-D workgroup dispatch  – one workgroup per output tile
5.  Thread indexing  – local_invocation_id vs global_invocation_id
6.  Performance considerations  – memory coalescing, occupancy, tile size

Background
----------
In the naive matmul (see compute_matmul.py) every thread independently reads
rows from A and columns from B.  When the matrices are large, each read is a
cache-miss that fetches data all the way from global GPU memory (VRAM).

Tiling exploits *data reuse*: a workgroup of TILE×TILE threads loads a TILE×TILE
sub-block of A and B into fast workgroup memory *once*, then all threads in the
workgroup reuse those values for TILE dot-product steps before loading the next
tile.  This can reduce global memory traffic by ~TILE×.

Run this file directly::

    python examples/compute_tiled_matmul.py
"""

import struct
import math
import numpy as np
import wgpu

# ---------------------------------------------------------------------------
# TILE SIZE – the side length of each square tile loaded into workgroup memory.
# Each workgroup launches TILE*TILE threads arranged in a 2-D grid.
# Larger tiles mean better data reuse but require more workgroup memory and
# limit how many workgroups can be live at once (occupancy trade-off).
# Common choices: 8, 16, 32.  We use 16 here.
# ---------------------------------------------------------------------------
TILE = 16

# ---------------------------------------------------------------------------
# MATRIX DIMENSIONS  (edit these to experiment)
# ---------------------------------------------------------------------------
# A is (M x K), B is (K x N), result C is (M x N).
# Values do NOT need to be multiples of TILE – the shader zero-pads the last
# tile when the dimension is not a multiple of TILE.
M = 64   # rows of A / rows of C
K = 48   # cols of A / rows of B
N = 32   # cols of B / cols of C

rng = np.random.default_rng(42)
A = rng.random((M, K), dtype=np.float32)
B = rng.random((K, N), dtype=np.float32)

# ---------------------------------------------------------------------------
# WGSL SHADER – TILED MATRIX MULTIPLICATION
# ---------------------------------------------------------------------------
# Key WGSL features used here:
#
#   var<workgroup>          – declares workgroup (shared) memory.
#                             All threads in the workgroup see the same array.
#                             It lives in on-chip SRAM (~tens of KB per SM).
#
#   @builtin(local_invocation_id)  – 3-D index of the thread *within* its
#                                    workgroup: (0..TILE-1, 0..TILE-1, 0).
#   @builtin(workgroup_id)         – 3-D index of the workgroup itself.
#   @builtin(global_invocation_id) – workgroup_id * workgroup_size + local_id.
#
#   workgroupBarrier()      – memory barrier: waits until every thread in the
#                             workgroup has finished writing to workgroup memory.
#                             Prevents any thread from reading a tile element
#                             before all threads have loaded it.
#
# Algorithm:
#   For each output tile (row-block, col-block):
#     For t = 0 .. ceil(K/TILE) - 1:          # iterate over K tiles
#       1. Load A[row, t*TILE .. (t+1)*TILE-1] into tile_A[local_row, local_col]
#       2. Load B[t*TILE .. (t+1)*TILE-1, col] into tile_B[local_row, local_col]
#       3. workgroupBarrier()  -- everyone must finish loading before anyone reads
#       4. Accumulate: for k in 0..TILE-1: acc += tile_A[local_row, k] * tile_B[k, local_col]
#       5. workgroupBarrier()  -- everyone must finish reading before anyone overwrites

shader_source = f"""
// ── Buffers ──────────────────────────────────────────────────────────────────
@group(0) @binding(0) var<storage, read>       mat_a      : array<f32>;
@group(0) @binding(1) var<storage, read>       mat_b      : array<f32>;
@group(0) @binding(2) var<storage, read_write> mat_c      : array<f32>;
@group(0) @binding(3) var<uniform>             dims       : vec4<u32>;
// dims.x = M (rows of A / rows of C)
// dims.y = K (cols of A / rows of B)
// dims.z = N (cols of B / cols of C)

// ── Workgroup (shared) memory ─────────────────────────────────────────────────
// Each workgroup loads one TILE×TILE sub-block of A and one of B.
// These arrays live in fast on-chip SRAM, shared among all TILE*TILE threads.
var<workgroup> tile_a: array<f32, {TILE * TILE}>;
var<workgroup> tile_b: array<f32, {TILE * TILE}>;

// ── Entry point ───────────────────────────────────────────────────────────────
@compute @workgroup_size({TILE}, {TILE}, 1)
fn main(
    @builtin(global_invocation_id) gid   : vec3<u32>,   // global thread id
    @builtin(local_invocation_id)  lid   : vec3<u32>,   // thread id within workgroup
) {{
    let row = gid.y;   // global row index in C (and A)
    let col = gid.x;   // global col index in C (and B)
    let lrow = lid.y;  // local row within tile (0..TILE-1)
    let lcol = lid.x;  // local col within tile (0..TILE-1)

    let M = dims.x;
    let K = dims.y;
    let N = dims.z;

    var acc: f32 = 0.0;

    // Number of tiles along the K dimension.
    let num_tiles = (K + {TILE}u - 1u) / {TILE}u;

    for (var t: u32 = 0u; t < num_tiles; t++) {{

        // ── Step 1: Load a tile of A into workgroup memory ────────────────────
        // Thread (lrow, lcol) loads A[row, t*TILE + lcol].
        let a_col = t * {TILE}u + lcol;
        if (row < M && a_col < K) {{
            tile_a[lrow * {TILE}u + lcol] = mat_a[row * K + a_col];
        }} else {{
            tile_a[lrow * {TILE}u + lcol] = 0.0;   // zero-pad out-of-bounds
        }}

        // ── Step 2: Load a tile of B into workgroup memory ────────────────────
        // Thread (lrow, lcol) loads B[t*TILE + lrow, col].
        let b_row = t * {TILE}u + lrow;
        if (b_row < K && col < N) {{
            tile_b[lrow * {TILE}u + lcol] = mat_b[b_row * N + col];
        }} else {{
            tile_b[lrow * {TILE}u + lcol] = 0.0;   // zero-pad out-of-bounds
        }}

        // ── Step 3: Barrier – wait for ALL threads to finish loading ──────────
        // Without this barrier a fast thread might start reading tile_a/tile_b
        // before a slow thread has written its element, giving wrong results.
        workgroupBarrier();

        // ── Step 4: Accumulate the partial dot-product ────────────────────────
        for (var k: u32 = 0u; k < {TILE}u; k++) {{
            acc += tile_a[lrow * {TILE}u + k] * tile_b[k * {TILE}u + lcol];
        }}

        // ── Step 5: Barrier – wait before overwriting tile on next iteration ──
        // Without this second barrier, a fast thread on iteration t+1 could
        // overwrite tile_a or tile_b while a slow thread is still reading them.
        workgroupBarrier();
    }}

    // Write the result only if this thread maps to a valid output element.
    if (row < M && col < N) {{
        mat_c[row * N + col] = acc;
    }}
}}
"""

# ---------------------------------------------------------------------------
# DEVICE SETUP
# ---------------------------------------------------------------------------
adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
device = adapter.request_device_sync()

# ---------------------------------------------------------------------------
# BUFFERS
# ---------------------------------------------------------------------------
# Flatten matrices to 1-D arrays (row-major / C order).
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

# Uniform buffer: pass M, K, N (and a padding word) as a vec4<u32>.
buf_dims = device.create_buffer_with_data(
    data=struct.pack("IIII", M, K, N, 0),
    usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
)

# ---------------------------------------------------------------------------
# PIPELINE SETUP
# ---------------------------------------------------------------------------
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
    {
        "binding": 3,
        "resource": {"buffer": buf_dims, "offset": 0, "size": buf_dims.size},
    },
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

# ---------------------------------------------------------------------------
# DISPATCH
# ---------------------------------------------------------------------------
# We dispatch a 2-D grid of workgroups.
# Each workgroup is TILE×TILE threads and covers a TILE×TILE output tile.
# Number of workgroups in X = ceil(N / TILE), in Y = ceil(M / TILE).
wg_x = math.ceil(N / TILE)
wg_y = math.ceil(M / TILE)

command_encoder = device.create_command_encoder()
compute_pass = command_encoder.begin_compute_pass()
compute_pass.set_pipeline(compute_pipeline)
compute_pass.set_bind_group(0, bind_group)
compute_pass.dispatch_workgroups(wg_x, wg_y, 1)
compute_pass.end()
device.queue.submit([command_encoder.finish()])

# ---------------------------------------------------------------------------
# VERIFY
# ---------------------------------------------------------------------------
raw = device.queue.read_buffer(buf_c)
C_gpu = np.frombuffer(raw, dtype=np.float32).reshape(M, N)
C_ref = A @ B

max_err = float(np.max(np.abs(C_gpu - C_ref)))
assert np.allclose(C_gpu, C_ref, atol=1e-4), f"Max error: {max_err}"
print(f"Tiled matmul ({M}x{K}) @ ({K}x{N}) = ({M}x{N}):  OK  (max_err={max_err:.2e})")

# ---------------------------------------------------------------------------
# EXERCISES
# ---------------------------------------------------------------------------
# 1. Change TILE to 8 or 32.  For large matrices (e.g. 512x512x512) profile
#    both with compute_timestamps.py as a template and compare throughput.
# 2. Remove the first workgroupBarrier() and run the shader.  Notice that
#    results are sometimes wrong (a race condition).
# 3. Remove the second workgroupBarrier() – can you construct a scenario
#    where results are wrong?
# 4. Extend the shader to handle rectangular tiles: TILE_M × TILE_N.
# 5. Implement a multi-pass pipeline: first compute C = A @ B, then
#    compute D = C + C^T using a second compute pass in the same
#    CommandEncoder.
