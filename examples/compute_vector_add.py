"""
GPU Compute Tutorial: Vector Addition in wgpu-py
=================================================

This example teaches GPU compute from first principles using wgpu-py.
It implements element-wise vector addition  C[i] = A[i] + B[i]  on the GPU.

Topics covered
--------------
1.  GPU execution model – threads, workgroups, warps
2.  wgpu concepts – adapters, devices, queues, pipelines, bind groups
3.  Compute-pipeline setup step-by-step
4.  WGSL shader introduction
5.  GPU memory types – storage, uniform, staging buffers
6.  Performance considerations – workgroup size, CPU-GPU sync
7.  Reading results back to the CPU
8.  Debugging hints

Run this file directly::

    python examples/compute_vector_add.py
"""

import wgpu

# ---------------------------------------------------------------------------
# 1.  GPU EXECUTION MODEL
# ---------------------------------------------------------------------------
# A GPU runs thousands of tiny threads simultaneously.  Threads are grouped
# into *workgroups* (also called thread-blocks in CUDA / work-groups in OpenCL).
# All threads within a workgroup share fast on-chip *workgroup memory* and can
# synchronise with each other via barriers.
#
# wgpu terminology:
#   - adapter   : a physical or virtual GPU (like a driver handle)
#   - device    : a logical connection to an adapter; the factory for all GPU
#                 objects (buffers, shaders, pipelines, …)
#   - queue     : the submission point for command buffers
#   - pipeline  : a compiled compute or render program
#   - bind group: a set of GPU resources (buffers, textures) bound to a shader

# ---------------------------------------------------------------------------
# 2.  ADAPTER AND DEVICE CREATION
# ---------------------------------------------------------------------------
# request_adapter_sync asks the underlying backend (Vulkan/Metal/DX12) for a
# GPU that satisfies our preferences.  "high-performance" picks a discrete GPU
# when one is present, otherwise the integrated GPU.
adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
print("Adapter:", adapter.summary)

# The device is the logical GPU connection.  All GPU objects are created from
# it.  The device also owns the command queue.
device = adapter.request_device_sync()

# You can list *all* available adapters (useful for multi-GPU systems):
#   for a in wgpu.gpu.enumerate_adapters_sync():
#       print(a.summary)

# ---------------------------------------------------------------------------
# 3.  WGSL COMPUTE SHADER
# ---------------------------------------------------------------------------
# WGSL (WebGPU Shading Language) is the shader language for WebGPU / wgpu.
#
# Key annotations:
#   @group(g) @binding(b)  – connects a shader variable to a bind-group slot
#   var<storage, read>     – read-only GPU storage buffer (like a const array)
#   var<storage, read_write> – writable GPU storage buffer
#   @compute               – marks the function as a compute entry point
#   @workgroup_size(x,y,z) – number of threads per workgroup in each dimension
#   @builtin(global_invocation_id) – the unique 3D thread index across all
#                                     workgroups; index.x is the flat index here
#
# Each invocation of `main` handles exactly ONE element of the arrays.
# We launch N workgroups of size (WORKGROUP_SIZE, 1, 1), so N*WORKGROUP_SIZE
# threads run in parallel.

# Threads per workgroup.  Must be a power of two for best GPU occupancy.
# GPUs execute threads in "warps" (NVIDIA) or "wavefronts" (AMD) of typically
# 32 or 64 threads.  Choosing a multiple of 64 keeps all hardware lanes busy.
WORKGROUP_SIZE = 64

shader_source = f"""
// Binding 0: input array A  (read-only storage buffer)
@group(0) @binding(0)
var<storage, read> a: array<f32>;

// Binding 1: input array B  (read-only storage buffer)
@group(0) @binding(1)
var<storage, read> b: array<f32>;

// Binding 2: output array C  (read-write storage buffer so the GPU can write)
@group(0) @binding(2)
var<storage, read_write> c: array<f32>;

// Uniform buffer that carries the array length so the shader knows when to
// stop.  Uniforms are read-only, small, and very fast to access because they
// live in a dedicated constant-cache on the GPU.
@group(0) @binding(3)
var<uniform> params: vec2<u32>;   // params.x == array length

// Compute entry point.
// @workgroup_size tells the GPU how many threads form one workgroup.
// Here we use a 1-D workgroup of {WORKGROUP_SIZE} threads.
@compute @workgroup_size({WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;          // flat thread index across ALL workgroups
    let n = params.x;       // array length

    // Guard: if we launched more threads than elements, skip the extras.
    // This happens when N is not a multiple of WORKGROUP_SIZE.
    if (i >= n) {{
        return;
    }}

    c[i] = a[i] + b[i];
}}
"""

# ---------------------------------------------------------------------------
# 4.  HOST DATA
# ---------------------------------------------------------------------------
# memoryview is a zero-copy view over raw bytes that supports the buffer
# protocol.  wgpu can consume anything that supports the buffer protocol
# (bytes, bytearray, memoryview, ctypes arrays, numpy arrays).

n = 1024  # number of floats

# Create two source vectors: a[i] = i, b[i] = i * 2.
data_a = memoryview(bytearray(n * 4)).cast("f")
data_b = memoryview(bytearray(n * 4)).cast("f")
for i in range(n):
    data_a[i] = float(i)
    data_b[i] = float(i * 2)

# ---------------------------------------------------------------------------
# 5.  GPU MEMORY – STORAGE BUFFERS AND UNIFORM BUFFERS
# ---------------------------------------------------------------------------
# GPU memory taxonomy:
#   storage buffer  – large, general-purpose read/write memory on the GPU.
#                     Accessed via @binding in WGSL.
#   uniform buffer  – small, read-only block of constants (transforms, sizes).
#                     Lives in a fast constant cache on the GPU.
#   staging buffer  – CPU-visible memory used to ferry data between CPU and GPU.
#                     wgpu hides this detail behind queue.write_buffer /
#                     queue.read_buffer.
#
# BufferUsage flags control what operations are valid on a buffer:
#   STORAGE  – bind as a storage buffer in a shader
#   UNIFORM  – bind as a uniform buffer in a shader
#   COPY_SRC – source of a copy operation (needed to read the result back)
#   COPY_DST – destination of a copy operation

# Input buffers: upload data immediately via create_buffer_with_data.
buf_a = device.create_buffer_with_data(data=data_a, usage=wgpu.BufferUsage.STORAGE)
buf_b = device.create_buffer_with_data(data=data_b, usage=wgpu.BufferUsage.STORAGE)

# Output buffer: just allocate space; the shader will fill it.
# COPY_SRC is required so we can later read the result back to the CPU.
buf_c = device.create_buffer(
    size=n * 4,
    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
)

# Uniform buffer: carry the array length to the shader.
# Use a 8-byte buffer for a vec2<u32> (two u32 values).
import struct

params_data = struct.pack("II", n, 0)  # length, padding
buf_params = device.create_buffer_with_data(
    data=params_data,
    usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
)

# ---------------------------------------------------------------------------
# 6.  BIND GROUP LAYOUT AND BIND GROUP
# ---------------------------------------------------------------------------
# A BindGroupLayout describes the *types* of resources a shader expects.
# A BindGroup is the *actual* resources (specific buffers) bound to those slots.
# Separating layout from data allows the same pipeline to work with different
# data by swapping bind groups – without recompiling the pipeline.

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
        "resource": {"buffer": buf_params, "offset": 0, "size": buf_params.size},
    },
]

bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
pipeline_layout = device.create_pipeline_layout(
    bind_group_layouts=[bind_group_layout]
)
bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

# ---------------------------------------------------------------------------
# 7.  SHADER MODULE AND COMPUTE PIPELINE
# ---------------------------------------------------------------------------
# The shader module is the compiled WGSL source.  The compute pipeline links
# the shader with its pipeline layout and entry point.
# Creating a pipeline triggers JIT compilation of the shader for the actual
# GPU hardware (Vulkan SPIR-V, Metal MSL, or DX12 DXBC/DXIL internally).

cshader = device.create_shader_module(code=shader_source)

compute_pipeline = device.create_compute_pipeline(
    layout=pipeline_layout,
    compute={"module": cshader, "entry_point": "main"},
)

# ---------------------------------------------------------------------------
# 8.  COMMAND ENCODER AND DISPATCH
# ---------------------------------------------------------------------------
# GPU commands are recorded into a CommandEncoder, then submitted as a batch.
# This allows the GPU driver to optimise the entire batch before execution.
#
# Dispatch arithmetic:
#   We want one thread per element.  With WORKGROUP_SIZE threads per workgroup
#   we need ceil(n / WORKGROUP_SIZE) workgroups.
#
#   dispatch_workgroups(x, y, z) launches x*y*z workgroups.
#   global_invocation_id.x ranges from 0 to x*WORKGROUP_SIZE - 1.

import math

num_workgroups = math.ceil(n / WORKGROUP_SIZE)

command_encoder = device.create_command_encoder()
compute_pass = command_encoder.begin_compute_pass()
compute_pass.set_pipeline(compute_pipeline)
compute_pass.set_bind_group(0, bind_group)
compute_pass.dispatch_workgroups(num_workgroups, 1, 1)
compute_pass.end()

# submit() sends the recorded commands to the GPU queue.
# The GPU executes them asynchronously; read_buffer (below) implicitly
# synchronises by waiting for completion before copying data back.
device.queue.submit([command_encoder.finish()])

# ---------------------------------------------------------------------------
# 9.  READ RESULTS BACK TO CPU (CPU-GPU SYNCHRONISATION)
# ---------------------------------------------------------------------------
# queue.read_buffer copies a GPU buffer to a CPU-visible memoryview.
# Internally wgpu creates a temporary "staging" (MAP_READ) buffer, issues a
# GPU copy command, waits for the GPU to finish, then maps the buffer and
# copies the bytes to Python.
#
# For high-throughput pipelines you would use async mapping and overlap GPU
# work with CPU work.  For simplicity, we use the synchronous helper here.

out = device.queue.read_buffer(buf_c).cast("f")
result = out.tolist()

# ---------------------------------------------------------------------------
# 10. VERIFY RESULTS
# ---------------------------------------------------------------------------
expected = [float(i) + float(i * 2) for i in range(n)]
assert result == expected, f"Mismatch! First bad index: {next(i for i,(r,e) in enumerate(zip(result, expected)) if r!=e)}"
print(f"Vector addition of {n} elements: OK")
print(f"  a[0..4]      = {[data_a[i] for i in range(4)]}")
print(f"  b[0..4]      = {[data_b[i] for i in range(4)]}")
print(f"  c[0..4]      = {result[:4]}")
print(f"  expected[0..4] = {expected[:4]}")

# ---------------------------------------------------------------------------
# EXERCISES
# ---------------------------------------------------------------------------
# 1. Change WORKGROUP_SIZE to 1, 32, 128, 256.  Observe any performance
#    difference (use compute_timestamps.py as a template for profiling).
# 2. Remove the bounds-guard in the shader (the `if (i >= n)` block) and
#    change n to a value that is NOT a multiple of WORKGROUP_SIZE.  What
#    happens?
# 3. Extend the shader to perform element-wise multiplication instead of
#    addition.
# 4. Add a second compute pass that squares the output (C[i] = C[i]*C[i])
#    in the same CommandEncoder.  Notice that no extra CPU-GPU sync is
#    needed between the two passes.
# 5. Replace the uniform buffer with an *override constant* in WGSL using
#    the `override` keyword and pass it via pipeline constants.
