"""
GPU Compute Tutorial: Parallel Reduction (Sum) in wgpu-py
==========================================================

This example teaches the *parallel reduction* pattern – one of the most
important GPU compute primitives.  We compute the sum of a large array on
the GPU using a two-pass approach.

Topics covered
--------------
1.  Why naive summation is slow on a GPU
2.  The butterfly/tree-reduction algorithm
3.  var<workgroup> scratch memory and workgroupBarrier()
4.  Two-pass reduction: per-workgroup partial sums → final total
5.  WGSL types: scalars, vectors, arrays, structs
6.  WGSL variable storage classes: private, function, workgroup, storage, uniform

Background
----------
A GPU has thousands of threads, but every thread must produce *one* output.
The naive approach (one thread loops over all N elements) is slower than a
single CPU core because all GPU parallelism is wasted.

The *parallel reduction* algorithm exploits all threads::

    Pass 1  (workgroup reduction):
    ┌───────────────────────────────────────────────┐
    │  WG 0:  thread 0..255 sum 256 elements → partial[0]  │
    │  WG 1:  thread 0..255 sum 256 elements → partial[1]  │
    │  …                                             │
    └───────────────────────────────────────────────┘
    Pass 2  (final reduction):
    ┌─────────────────────────────────────────────────┐
    │  WG 0:  thread 0..255 sum all partials → result[0]   │
    └─────────────────────────────────────────────────┘

Within each workgroup the reduction is a logarithmic *tree*::

    Step 0  (stride=128):
        thread 0 adds scratch[0] + scratch[128]
        thread 1 adds scratch[1] + scratch[129]
        ...
        thread 127 adds scratch[127] + scratch[255]
    ── workgroupBarrier() ──
    Step 1  (stride=64):
        thread 0 adds scratch[0] + scratch[64]
        ...
        thread 63 adds scratch[63] + scratch[127]
    ── workgroupBarrier() ──
    ... (log2(256) = 8 steps total)
    thread 0 now holds the sum for the entire workgroup.

Run this file directly::

    python examples/compute_reduction.py
"""

import array as arr
import math
import struct

import wgpu

# ---------------------------------------------------------------------------
# WGSL FUNDAMENTALS – used throughout this example
# ---------------------------------------------------------------------------
# WGSL Scalar types:
#   f32   – 32-bit float            (IEEE 754 single precision)
#   i32   – 32-bit signed integer
#   u32   – 32-bit unsigned integer
#   bool  – boolean
#
# WGSL Vector types (element count 2, 3, or 4):
#   vec2<f32>, vec3<f32>, vec4<f32>
#   vec2<u32>, vec3<u32>, vec4<u32>
#   Literal construction: vec3<f32>(1.0, 2.0, 3.0)
#   Swizzle:  v.x, v.y, v.z, v.w   (or .r, .g, .b, .a for colour)
#
# WGSL Matrix types (columns × rows):
#   mat2x2<f32>, mat3x3<f32>, mat4x4<f32>
#   mat4x4<f32>(…)  – column-major constructor
#
# WGSL Array types:
#   array<f32, 256>  – fixed-size array (required for var<workgroup>)
#   array<f32>       – runtime-sized array (only allowed in storage buffers)
#
# WGSL Struct types (user-defined):
#   struct MyParams { n: u32, pad: u32 }
#
# WGSL Variable storage classes:
#   var<storage, read>        – GPU storage buffer, read-only
#   var<storage, read_write>  – GPU storage buffer, writable
#   var<uniform>              – small read-only constants (constant cache)
#   var<workgroup>            – on-chip shared memory within one workgroup
#   var<private>              – thread-private register memory
#   var (inside a function)   – function-scope variable (same as private)
#
# WGSL Entry points:
#   @compute @workgroup_size(x, y, z) fn name(...) { ... }
#   @vertex fn name(...) -> @builtin(position) vec4<f32> { ... }
#   @fragment fn name(...) -> @location(0) vec4<f32> { ... }
#
# For compute shaders we only use @compute.
# ---------------------------------------------------------------------------

WORKGROUP_SIZE = 256  # threads per workgroup; must be a power of two

# ---------------------------------------------------------------------------
# SHADER PASS 1 – per-workgroup reduction
# ---------------------------------------------------------------------------
# We use a struct in the WGSL uniform to show how structs work in WGSL.
# The Python side packs the matching layout with struct.pack().
#
# IMPORTANT: each pass has its OWN shader module.  Putting both passes in a
# single module with the same @group/@binding numbers would be a WGSL
# validation error, because both entry points share the same compute stage.
# Separate modules keep binding namespaces cleanly independent.

shader_pass1 = f"""
// ─── Struct ──────────────────────────────────────────────────────────────────
// Structs group related fields.  The host must respect WGSL alignment rules:
//   u32 → 4-byte aligned, vec2<u32> → 8-byte, vec4 → 16-byte.
struct Params {{
    n          : u32,   // total number of elements in `data`
    n_partials : u32,   // number of partial-sum workgroups (ceil(n / {WORKGROUP_SIZE}))
}}

// ─── Bindings ────────────────────────────────────────────────────────────────
// @group(g) @binding(b) connects a WGSL variable to a host-side buffer.
// Pass 1 reads from `data` and writes per-workgroup partial sums to `partials`.
@group(0) @binding(0) var<storage, read>       data     : array<f32>;
@group(0) @binding(1) var<storage, read_write> partials : array<f32>;
@group(0) @binding(2) var<uniform>             params   : Params;

// ─── Workgroup (shared) memory ───────────────────────────────────────────────
// var<workgroup> is on-chip SRAM shared by all threads in one workgroup.
// It must be a *fixed-size* array (not runtime-sized).
// All {WORKGROUP_SIZE} threads in the workgroup see the SAME scratch[] array.
// Each new workgroup execution starts with uninitialised scratch memory.
var<workgroup> scratch: array<f32, {WORKGROUP_SIZE}>;

// ─── Pass 1 entry point ──────────────────────────────────────────────────────
// Each workgroup reduces {WORKGROUP_SIZE} consecutive elements down to ONE
// partial sum, which it writes to partials[workgroup_id.x].
@compute @workgroup_size({WORKGROUP_SIZE})
fn reduce_pass1(
    @builtin(global_invocation_id) gid : vec3<u32>,   // unique global thread id
    @builtin(local_invocation_id)  lid : vec3<u32>,   // id within this workgroup
    @builtin(workgroup_id)         wid : vec3<u32>,   // which workgroup
) {{
    let i         = gid.x;     // global element index this thread handles
    let local_idx = lid.x;     // 0 .. {WORKGROUP_SIZE}-1
    let n         = params.n;  // total elements (read from uniform)

    // ── Step 1: load one element into workgroup memory ──────────────────────
    // Guard: if i >= n (padding thread), load 0 so it does not corrupt the sum.
    // `select(false_val, true_val, condition)` is WGSL's ternary expression.
    scratch[local_idx] = select(0.0, data[i], i < n);

    // ── Step 2: barrier ─────────────────────────────────────────────────────
    // Wait until EVERY thread in the workgroup has written its element.
    // Without this barrier a fast thread might read scratch[local_idx+stride]
    // before a slow thread has written its value – a data race.
    workgroupBarrier();

    // ── Step 3: logarithmic tree reduction ──────────────────────────────────
    // Each iteration halves the active thread count.
    // After log₂({WORKGROUP_SIZE}) = {int(math.log2(WORKGROUP_SIZE))} steps,
    // scratch[0] holds the sum for the entire workgroup.
    //
    //   stride={WORKGROUP_SIZE // 2}: threads 0..{WORKGROUP_SIZE // 2 - 1} each add one peer
    //   stride={WORKGROUP_SIZE // 4}: threads 0..{WORKGROUP_SIZE // 4 - 1} each add one peer
    //   ...
    //   stride=1: only thread 0 adds scratch[0] + scratch[1]
    var stride: u32 = {WORKGROUP_SIZE}u >> 1u;   // initial stride = {WORKGROUP_SIZE // 2}
    loop {{
        if (stride == 0u) {{ break; }}
        if (local_idx < stride) {{
            scratch[local_idx] += scratch[local_idx + stride];
        }}
        // All threads must finish reading AND writing before the next round.
        workgroupBarrier();
        stride = stride >> 1u;   // stride /= 2
    }}

    // ── Step 4: write partial sum ────────────────────────────────────────────
    // Only thread 0 of each workgroup writes the workgroup's sum.
    if (local_idx == 0u) {{
        partials[wid.x] = scratch[0];
    }}
}}
"""

# ---------------------------------------------------------------------------
# SHADER PASS 2 – reduce partial sums to final total
# ---------------------------------------------------------------------------
# A single workgroup reduces all partial sums from pass 1.
# The binding layout is identical to pass 1 (same slot numbers) but points
# at different buffers via the host-side bind group.

shader_pass2 = f"""
struct Params {{
    n          : u32,
    n_partials : u32,
}}

// Pass 2 bindings: `partials` (from pass 1) → read; `result` → write.
@group(0) @binding(0) var<storage, read>       partials : array<f32>;
@group(0) @binding(1) var<storage, read_write> result   : array<f32>;
@group(0) @binding(2) var<uniform>             params   : Params;

// Same workgroup scratch buffer pattern as pass 1.
var<workgroup> scratch: array<f32, {WORKGROUP_SIZE}>;

// Pass 2 launches exactly ONE workgroup.
// Constraint: n_partials <= {WORKGROUP_SIZE}
// (satisfied when N <= {WORKGROUP_SIZE}² = {WORKGROUP_SIZE**2}).
@compute @workgroup_size({WORKGROUP_SIZE})
fn reduce_pass2(
    @builtin(local_invocation_id) lid : vec3<u32>,
) {{
    let local_idx  = lid.x;
    let n_partials = params.n_partials;

    // Load one partial sum (or 0 if out of range).
    scratch[local_idx] = select(0.0, partials[local_idx], local_idx < n_partials);
    workgroupBarrier();

    // Identical tree reduction as pass 1.
    var stride: u32 = {WORKGROUP_SIZE}u >> 1u;
    loop {{
        if (stride == 0u) {{ break; }}
        if (local_idx < stride) {{
            scratch[local_idx] += scratch[local_idx + stride];
        }}
        workgroupBarrier();
        stride = stride >> 1u;
    }}

    // result[0] is the final global sum.
    if (local_idx == 0u) {{
        result[0] = scratch[0];
    }}
}}
"""

# ---------------------------------------------------------------------------
# WGSL VARIABLE STORAGE CLASS RECAP
# ---------------------------------------------------------------------------
# The shaders above illustrate all five storage classes:
#
#   var<storage, read>       data, partials  (pass 2 read side)
#       Lives in GPU VRAM.  Read-only from the shader.
#       Wired to a host buffer via @group/@binding.
#
#   var<storage, read_write> partials (pass 1 write side), result
#       Lives in GPU VRAM.  Readable and writable from the shader.
#       The host buffer needs wgpu.BufferUsage.STORAGE.
#       Add COPY_SRC to read the buffer back to the CPU.
#
#   var<uniform>             params
#       Small (≤64 KB), read-only constants.  Backed by a dedicated GPU
#       constant cache.  Must be set by @group/@binding.
#
#   var<workgroup>           scratch
#       On-chip SRAM shared within one workgroup only.
#       NOT visible between workgroups.  Fixed size at compile time.
#       Each new workgroup invocation starts with uninitialised contents.
#
#   var<private>             (not in this example, but common for helpers)
#       Per-thread register memory, private to each invocation.
#       Example:  var<private> accumulator: f32 = 0.0;
#
#   var (function scope)     e.g., `var stride: u32 = ...`
#       Equivalent to var<function>: each invocation has its own copy.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# HOST SETUP
# ---------------------------------------------------------------------------
adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
print("Adapter:", adapter.summary)
device = adapter.request_device_sync()

# Number of floats to reduce.  Deliberately not a multiple of WORKGROUP_SIZE
# to exercise the bounds-guard logic in the shader.
N = 1000

# Build input data: [1.0, 1.0, ..., 1.0] for easy verification (sum == N).
data_host = arr.array("f", [1.0] * N)

# ---------------------------------------------------------------------------
# BUFFERS
# ---------------------------------------------------------------------------
# Number of workgroups = ceil(N / WORKGROUP_SIZE).
# Each workgroup writes one partial sum, so the partials buffer has
# n_workgroups elements.
n_workgroups = math.ceil(N / WORKGROUP_SIZE)

# data: input array uploaded to GPU storage.
buf_data = device.create_buffer_with_data(
    data=data_host,
    usage=wgpu.BufferUsage.STORAGE,
)

# partials: output of pass 1, input to pass 2.
# We need COPY_SRC only for debugging (to inspect partial sums from Python).
buf_partials = device.create_buffer(
    size=n_workgroups * 4,
    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
)

# result: single f32 written by pass 2.
# COPY_SRC is required to read it back to the CPU with queue.read_buffer().
buf_result = device.create_buffer(
    size=4,  # one f32
    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
)

# Uniform: struct Params { n: u32, n_partials: u32 }
# Both passes share the same uniform buffer.
# struct.pack("II", …) produces two little-endian uint32 values.
params_data = struct.pack("II", N, n_workgroups)
buf_params = device.create_buffer_with_data(
    data=params_data,
    usage=wgpu.BufferUsage.UNIFORM,
)

# ---------------------------------------------------------------------------
# SHADER MODULES – one per pass
# ---------------------------------------------------------------------------
# Each shader module is compiled independently, so binding @group(0)@binding(0)
# can mean different things in each module without conflict.
cshader1 = device.create_shader_module(code=shader_pass1)
cshader2 = device.create_shader_module(code=shader_pass2)

# ---------------------------------------------------------------------------
# BIND GROUP LAYOUT – shared template for both passes
# ---------------------------------------------------------------------------
# Both passes use the same layout shape:
#   binding 0 → read_only_storage
#   binding 1 → storage (read_write)
#   binding 2 → uniform
# This lets us reuse the same layout object for both pipelines.

shared_bgl = device.create_bind_group_layout(
    entries=[
        {
            "binding": 0,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
        },
        {
            "binding": 1,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {"type": wgpu.BufferBindingType.storage},
        },
        {
            "binding": 2,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {"type": wgpu.BufferBindingType.uniform},
        },
    ]
)
shared_layout = device.create_pipeline_layout(bind_group_layouts=[shared_bgl])

# ---------------------------------------------------------------------------
# PIPELINES
# ---------------------------------------------------------------------------
pipeline1 = device.create_compute_pipeline(
    layout=shared_layout,
    compute={"module": cshader1, "entry_point": "reduce_pass1"},
)
pipeline2 = device.create_compute_pipeline(
    layout=shared_layout,
    compute={"module": cshader2, "entry_point": "reduce_pass2"},
)

# ---------------------------------------------------------------------------
# BIND GROUPS – wire actual buffers to the layout slots
# ---------------------------------------------------------------------------
# Pass 1: data (read) → partials (write) → params
bind_group1 = device.create_bind_group(
    layout=shared_bgl,
    entries=[
        {"binding": 0, "resource": {"buffer": buf_data,     "offset": 0, "size": buf_data.size}},
        {"binding": 1, "resource": {"buffer": buf_partials, "offset": 0, "size": buf_partials.size}},
        {"binding": 2, "resource": {"buffer": buf_params,   "offset": 0, "size": buf_params.size}},
    ],
)

# Pass 2: partials (read) → result (write) → params
bind_group2 = device.create_bind_group(
    layout=shared_bgl,
    entries=[
        {"binding": 0, "resource": {"buffer": buf_partials, "offset": 0, "size": buf_partials.size}},
        {"binding": 1, "resource": {"buffer": buf_result,   "offset": 0, "size": buf_result.size}},
        {"binding": 2, "resource": {"buffer": buf_params,   "offset": 0, "size": buf_params.size}},
    ],
)

# ---------------------------------------------------------------------------
# DISPATCH
# ---------------------------------------------------------------------------
# Both passes are recorded into a SINGLE CommandEncoder.
# wgpu inserts a full pipeline barrier between consecutive passes
# automatically, so pass 2 is guaranteed to see all writes from pass 1.

command_encoder = device.create_command_encoder()

# Pass 1: n_workgroups workgroups, each of WORKGROUP_SIZE threads.
pass1 = command_encoder.begin_compute_pass()
pass1.set_pipeline(pipeline1)
pass1.set_bind_group(0, bind_group1)
pass1.dispatch_workgroups(n_workgroups, 1, 1)
pass1.end()

# Pass 2: ONE workgroup reduces all n_workgroups partial sums.
# Requires n_workgroups <= WORKGROUP_SIZE, i.e. N <= WORKGROUP_SIZE**2.
pass2 = command_encoder.begin_compute_pass()
pass2.set_pipeline(pipeline2)
pass2.set_bind_group(0, bind_group2)
pass2.dispatch_workgroups(1, 1, 1)
pass2.end()

device.queue.submit([command_encoder.finish()])

# ---------------------------------------------------------------------------
# READ RESULTS
# ---------------------------------------------------------------------------
# queue.read_buffer() waits for the GPU to finish before copying bytes to CPU.
result_bytes = device.queue.read_buffer(buf_result)
gpu_sum = struct.unpack("f", result_bytes)[0]

cpu_sum = float(sum(data_host))
print(f"N = {N}, WORKGROUP_SIZE = {WORKGROUP_SIZE}, n_workgroups = {n_workgroups}")
print(f"  GPU sum = {gpu_sum}")
print(f"  CPU sum = {cpu_sum}")

# f32 has ~7 decimal digits of precision; allow small rounding error.
assert abs(gpu_sum - cpu_sum) < 0.01 * abs(cpu_sum) + 1e-3, (
    f"Reduction mismatch: gpu={gpu_sum}, cpu={cpu_sum}"
)
print("Parallel reduction: OK")

# ---------------------------------------------------------------------------
# EXERCISES
# ---------------------------------------------------------------------------
# 1. Change N to values that ARE and ARE NOT multiples of WORKGROUP_SIZE.
#    Confirm the result is always correct (the select() guard handles padding).
#
# 2. Remove the first workgroupBarrier() in reduce_pass1 (after loading
#    scratch[]).  Run many times – do you ever get wrong results?
#
# 3. Change the input from all-ones to a ramp: data_host[i] = float(i+1).
#    The expected sum is N*(N+1)//2.  Verify.
#
# 4. Implement a parallel MAXIMUM (replace `+=` with `max()`).
#
# 5. Implement the reduction using atomicAdd() on an atomic<u32> or atomic<i32>
#    result buffer in a single pass.  Compare performance with the two-pass
#    version for large N.
#
# 6. If N > WORKGROUP_SIZE**2 (= 65536), a third pass is needed.
#    Extend the example to support arbitrarily large N iteratively.
