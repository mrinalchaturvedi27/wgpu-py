GPU Compute Guide
=================

This guide teaches GPU compute in **wgpu-py** from first principles.
All concepts are illustrated with runnable examples from the repository.


.. contents:: Contents
   :local:
   :depth: 2


1. GPU Compute Fundamentals
---------------------------

Threads, Workgroups, and Warps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A GPU is a massively parallel processor.  Unlike a CPU – which has a handful of
powerful cores – a modern GPU has thousands of small cores that run thousands of
threads simultaneously.

Threads are organised in a two-level hierarchy:

* **Thread** – the smallest unit of execution.  Each thread runs the same shader
  program (SPMD – Single Program, Multiple Data) but with a different *invocation
  index*.
* **Workgroup** (= *thread-block* in CUDA, *work-group* in OpenCL) – a fixed-size
  group of threads that run on the same *streaming multiprocessor* (SM / CU) and
  share on-chip *workgroup memory*.  Threads within a workgroup can synchronise
  with each other using barriers.
* **Warp / Wavefront** – the hardware scheduling unit.  NVIDIA GPUs execute
  threads in groups of 32 (warps); AMD in groups of 64 (wavefronts).  If your
  workgroup size is not a multiple of the warp size, some hardware lanes are idle
  and you waste occupancy.

Memory hierarchy (fastest → slowest, smallest → largest):

1. **Registers** – private per-thread, ~tens of 32-bit values.
2. **Workgroup (shared) memory** – on-chip SRAM shared within a workgroup,
   ~32–96 KB per SM.  Access latency ≈ 1–5 ns.
3. **L1 / L2 cache** – managed automatically by the GPU.
4. **Global (device) memory** – off-chip GDDR/HBM VRAM, hundreds of GB/s
   bandwidth but ≈ 100–200 ns latency.
5. **Host (CPU) memory** – accessible by the GPU only via PCIe transfers.


wgpu Terminology
~~~~~~~~~~~~~~~~

+-----------------------------+-------------------------------------------------------+
| wgpu concept                | Maps to ...                                           |
+=============================+=======================================================+
| ``GPUAdapter``              | A physical or virtual GPU (and its driver).           |
+-----------------------------+-------------------------------------------------------+
| ``GPUDevice``               | A logical connection to an adapter; the factory for   |
|                             | all GPU objects (buffers, textures, pipelines, ...).  |
+-----------------------------+-------------------------------------------------------+
| ``GPUQueue``                | The submission point for command buffers.             |
+-----------------------------+-------------------------------------------------------+
| ``GPUComputePipeline``      | A compiled compute program (shader + layout).         |
+-----------------------------+-------------------------------------------------------+
| ``GPUBindGroup``            | A set of buffers/textures wired to shader bindings.   |
+-----------------------------+-------------------------------------------------------+
| ``GPUCommandEncoder``       | Records a batch of GPU commands before submission.    |
+-----------------------------+-------------------------------------------------------+
| ``GPUComputePassEncoder``   | Records compute commands within a command buffer.     |
+-----------------------------+-------------------------------------------------------+

**Exercise 1:** List all adapters on your machine and print their summaries:

.. code-block:: python

    import wgpu
    for a in wgpu.gpu.enumerate_adapters_sync():
        print(a.summary)


2. wgpu Architecture
--------------------

How wgpu Abstracts the Backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

wgpu-py is a Python wrapper around `wgpu-native
<https://github.com/gfx-rs/wgpu-native>`_, which is a C-API wrapper around the
`wgpu <https://github.com/gfx-rs/wgpu>`_ Rust crate.  The Rust crate implements
the `WebGPU specification <https://gpuweb.github.io/gpuweb/>`_ on top of the
native graphics APIs:

.. code-block:: text

    Python application
        │
        ▼
    wgpu-py  (this library, pure Python + ctypes)
        │
        ▼
    wgpu-native  (C FFI layer, auto-generated from wgpu Rust crate)
        │
        ▼
    wgpu (Rust)   ──►  Vulkan  (Linux, Windows, Android)
                  ──►  Metal   (macOS, iOS)
                  ──►  DX12    (Windows)
                  ──►  OpenGL  (fallback)

You can force a specific backend with the environment variable
``WGPU_BACKEND_TYPE`` (values: ``"Vulkan"``, ``"Metal"``, ``"D3D12"``,
``"OpenGL"``).


Device and Queue Creation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import wgpu

    # 1. Pick an adapter (abstract GPU)
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")

    # 2. Create a logical device and its queue
    device = adapter.request_device_sync()
    queue  = device.queue      # GPUQueue – the submission point

Internally, ``request_device_sync`` calls through to the Rust backend which
opens a Vulkan/Metal/DX12 device and allocates a command queue.


Command Buffers and Command Encoders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPU commands are not executed immediately; they are *recorded* into a
``GPUCommandEncoder`` and submitted as a batch.  This design lets the GPU driver
optimise the whole batch (pipeline barriers, memory aliasing, etc.) before
execution begins.

.. code-block:: python

    encoder = device.create_command_encoder()

    # Begin a compute pass
    pass_ = encoder.begin_compute_pass()
    pass_.set_pipeline(pipeline)
    pass_.set_bind_group(0, bind_group)
    pass_.dispatch_workgroups(nx, ny, nz)
    pass_.end()

    # Finish recording → produces a GPUCommandBuffer
    cmd_buf = encoder.finish()

    # Submit to the GPU queue
    device.queue.submit([cmd_buf])

Multiple passes (compute or render) can be recorded into a single encoder.
They execute on the GPU in the order they were recorded.

**Exercise 2:** Modify ``examples/compute_noop.py`` to record two consecutive
compute passes in the same command encoder.  Confirm that the second pass sees
the output of the first.


3. Compute Pipeline
-------------------

Building a Compute Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A compute pipeline ties together a shader module, its entry-point name, and a
pipeline layout that describes all resource bindings.

.. code-block:: python

    # 1. Compile the shader
    shader = device.create_shader_module(code=wgsl_source)

    # 2. Describe what resources the shader needs
    bgl = device.create_bind_group_layout(entries=[
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
    ])

    # 3. Create the pipeline layout (can include multiple bind group layouts)
    layout = device.create_pipeline_layout(bind_group_layouts=[bgl])

    # 4. Compile the pipeline (shader specialisation happens here)
    pipeline = device.create_compute_pipeline(
        layout=layout,
        compute={"module": shader, "entry_point": "main"},
    )

Alternatively, use ``layout="auto"`` to let wgpu infer the layout from the
shader reflection data:

.. code-block:: python

    pipeline = device.create_compute_pipeline(
        layout="auto",
        compute={"module": shader, "entry_point": "main"},
    )
    bgl = pipeline.get_bind_group_layout(0)

Bind Groups
~~~~~~~~~~~

A ``GPUBindGroup`` wires actual GPU resources to the abstract slots described by
the layout:

.. code-block:: python

    bind_group = device.create_bind_group(
        layout=bgl,
        entries=[
            {"binding": 0, "resource": {"buffer": buf_in,  "offset": 0, "size": buf_in.size}},
            {"binding": 1, "resource": {"buffer": buf_out, "offset": 0, "size": buf_out.size}},
        ],
    )

Dispatching Workgroups
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import math

    # Launch enough workgroups so every element is covered.
    # workgroup_size must match @workgroup_size in WGSL.
    workgroup_size = 64
    n_workgroups = math.ceil(n / workgroup_size)

    pass_.dispatch_workgroups(n_workgroups, 1, 1)

For a 2-D problem (e.g. matrix), dispatch in two dimensions:

.. code-block:: python

    pass_.dispatch_workgroups(math.ceil(N/TILE), math.ceil(M/TILE), 1)

See ``examples/compute_vector_add.py`` for the full 1-D pipeline and
``examples/compute_tiled_matmul.py`` for the full 2-D pipeline.

**Exercise 3:** Open ``examples/compute_noop.py``.  Trace every object created
(shader module → bind group layout → pipeline layout → bind group → pipeline →
command encoder → compute pass) and draw a dependency diagram.


4. WGSL Compute Shaders
-----------------------

Storage Buffers
~~~~~~~~~~~~~~~~

Storage buffers are the primary way to pass large arrays to a shader.

.. code-block:: wgsl

    @group(0) @binding(0) var<storage, read>       input  : array<f32>;
    @group(0) @binding(1) var<storage, read_write> output : array<f32>;

* ``read`` – read-only; the GPU can keep the data in a faster cache.
* ``read_write`` – writable; the GPU must treat it as coherent memory.

Uniform Buffers
~~~~~~~~~~~~~~~~

Uniform buffers carry small, constant data (parameters, transforms).  They are
read-only and live in a fast constant cache:

.. code-block:: wgsl

    struct Params { n: u32, stride: u32 }
    @group(0) @binding(2) var<uniform> params: Params;

Workgroup Memory
~~~~~~~~~~~~~~~~

Workgroup (shared) memory is on-chip SRAM shared by all threads in a workgroup.
Declare it at module scope:

.. code-block:: wgsl

    var<workgroup> tile: array<f32, 256>;   // 256 floats of shared memory

All threads in the workgroup read and write the same ``tile`` array.  It is much
faster than global memory but limited in size (~32–96 KB per SM).

Thread Indexing
~~~~~~~~~~~~~~~~

Three built-in 3-D indices are available in a compute shader:

.. code-block:: wgsl

    @builtin(global_invocation_id) gid : vec3<u32>
    // Unique thread id across ALL workgroups.
    // gid = workgroup_id * workgroup_size + local_invocation_id

    @builtin(local_invocation_id) lid : vec3<u32>
    // Thread id *within* its workgroup: 0 .. workgroup_size-1

    @builtin(workgroup_id) wid : vec3<u32>
    // Which workgroup this thread belongs to.

Synchronisation (Barriers)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: wgsl

    workgroupBarrier();

``workgroupBarrier()`` ensures:

1. All threads in the workgroup have reached this point.
2. All workgroup-memory writes made before the barrier are visible to all
   threads after it.

Without barriers, threads can race: one thread might read a tile element before
another thread has written it.

See ``examples/compute_tiled_matmul.py`` for a complete example with two
``workgroupBarrier()`` calls – one after loading tiles, one before
overwriting them.

**Exercise 4:** Write a WGSL shader that computes the prefix sum (cumulative
sum) of an array using workgroup memory.  Hint: the parallel prefix-sum
algorithm needs exactly ``log2(WORKGROUP_SIZE)`` barrier-separated passes.


5. Memory and Performance
--------------------------

GPU Memory Types Summary
~~~~~~~~~~~~~~~~~~~~~~~~~

+--------------------------+----------+------------+-----------------------------+
| Memory type              | Location | Access     | wgpu usage flags            |
+==========================+==========+============+=============================+
| Storage buffer           | VRAM     | shader R/W | ``STORAGE``                 |
+--------------------------+----------+------------+-----------------------------+
| Uniform buffer           | VRAM     | shader R   | ``UNIFORM``                 |
+--------------------------+----------+------------+-----------------------------+
| Workgroup memory         | On-chip  | shader R/W | ``var<workgroup>`` in WGSL  |
+--------------------------+----------+------------+-----------------------------+
| Staging buffer           | CPU-vis  | CPU read   | ``MAP_READ`` / ``COPY_DST`` |
+--------------------------+----------+------------+-----------------------------+
| Vertex/index buffer      | VRAM     | vertex     | ``VERTEX`` / ``INDEX``      |
+--------------------------+----------+------------+-----------------------------+

**Staging buffers** are the mechanism for CPU↔GPU data transfer.  When you call
``device.queue.read_buffer(buf)``, wgpu internally:

1. Allocates a ``MAP_READ | COPY_DST`` staging buffer.
2. Records a ``copy_buffer_to_buffer`` command.
3. Submits and waits for the GPU to finish.
4. Maps the staging buffer into CPU address space.
5. Copies the bytes to a Python ``memoryview``.

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Coalesced memory access**
    Threads in the same warp should access *consecutive* memory addresses.  If
    thread 0 reads ``data[0]``, thread 1 reads ``data[1]``, … the hardware
    combines all reads into a single wide memory transaction.  Strided access
    (thread 0 reads ``data[0]``, thread 1 reads ``data[64]``, …) issues
    separate transactions and is much slower.

**Occupancy**
    Occupancy is the ratio of active warps to the maximum number of warps an SM
    can hold.  Low occupancy leaves ALU lanes idle.  Main limiters:
    - Workgroup memory usage (too much → fewer workgroups per SM).
    - Register usage per thread (too many → fewer threads per SM).
    - Workgroup size (must be a multiple of the warp size, typically 32 or 64).

**Workgroup size**
    A common heuristic is 64 or 256 for 1-D problems.  For 2-D tile problems,
    use TILE × TILE (e.g. 16 × 16 = 256).  Always profile; the ideal size is
    hardware-dependent.

**Minimising CPU-GPU synchronisation**
    Every ``read_buffer`` call stalls the CPU until the GPU finishes.  Where
    possible, batch multiple dispatch calls before reading results, or use async
    buffer mapping to overlap CPU and GPU work.

**Exercise 5:** Profile the vector-addition example with WORKGROUP_SIZE =
1, 64, 256 using ``examples/compute_timestamps.py`` as a template for GPU
timing.


6. Build a Compute Program
--------------------------

A. Vector Addition
~~~~~~~~~~~~~~~~~~~

See ``examples/compute_vector_add.py`` for the full, annotated implementation.

Key steps:

1. Request an adapter and device.
2. Write the WGSL shader (storage buffers + uniform for the length).
3. Create GPU buffers for inputs, output, and uniforms.
4. Build the bind group layout, pipeline layout, bind group, and pipeline.
5. Record a ``dispatch_workgroups`` call in a command encoder.
6. Submit and read results back with ``queue.read_buffer``.


B. Naive Matrix Multiplication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See ``examples/compute_matmul.py`` for the existing naive implementation.

Each thread computes one element of the output matrix:

.. code-block:: wgsl

    @compute @workgroup_size(1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let row = gid.y;
        let col = gid.x;
        var acc: f32 = 0.0;
        for (var k: u32 = 0u; k < K; k++) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }

Dispatch: ``dispatch_workgroups(N, M, 1)`` — one workgroup per output element.

The bottleneck is global-memory bandwidth: each thread re-reads the same rows
and columns of A and B over and over from slow VRAM.


C. Tiled Matrix Multiplication with Shared Memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See ``examples/compute_tiled_matmul.py`` for the full implementation.

The key idea: instead of one thread per element, use one *workgroup* (of TILE×TILE
threads) per output *tile*.  The workgroup collectively loads a TILE×TILE block
of A and B into fast workgroup memory, computes partial dot-products, then
repeats for the next block along the K dimension.

.. code-block:: wgsl

    var<workgroup> tile_a: array<f32, TILE * TILE>;
    var<workgroup> tile_b: array<f32, TILE * TILE>;

    // For each K-tile:
    //   1. All threads cooperatively load a tile of A and B
    //   2. workgroupBarrier()
    //   3. Accumulate: for k in 0..TILE { acc += tile_a[lrow,k] * tile_b[k,lcol] }
    //   4. workgroupBarrier()

Global memory reads per element: reduced from O(K) to O(K / TILE) — a TILE×
speedup in memory traffic.

**Exercise 6:** Implement the multi-pass version: after computing C = A @ B,
add a second compute pass that element-wise multiplies C by a scaling vector
stored in a separate buffer.  Verify against NumPy.


7. Debugging and Profiling
--------------------------

Validation Layers
~~~~~~~~~~~~~~~~~~

wgpu includes built-in validation that checks API usage.  When validation fails
you will see Python exceptions with descriptive messages such as
``"Buffer is not large enough"`` or ``"Binding 0 missing from bind group"``.

To get more verbose Rust-level logs, set the Python log level:

.. code-block:: python

    import logging
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("wgpu").setLevel(logging.DEBUG)

Vulkan validation layers (the most thorough option) are enabled when you:

1. Install the `LunarG Vulkan SDK <https://vulkan.lunarg.com/>`_.
2. Build wgpu-native in *debug* mode (or use a debug wheel).
3. Set ``WGPU_BACKEND_TYPE=Vulkan``.


Object Labels and Debug Markers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every GPU object can be given a human-readable label that appears in error
messages and profiling tools:

.. code-block:: python

    buf = device.create_buffer(label="input-A", size=..., usage=...)
    pipeline = device.create_compute_pipeline(
        label="matmul-pipeline", layout=layout, compute=...
    )

Inside a compute pass you can insert debug groups:

.. code-block:: python

    pass_.push_debug_group("tiled-matmul")
    pass_.dispatch_workgroups(wg_x, wg_y, 1)
    pass_.pop_debug_group()


GPU Timestamp Queries
~~~~~~~~~~~~~~~~~~~~~~

Use ``FeatureName.timestamp_query`` to measure GPU-side execution time:

.. code-block:: python

    device = adapter.request_device_sync(
        required_features=[wgpu.FeatureName.timestamp_query]
    )
    query_set = device.create_query_set(type=wgpu.QueryType.timestamp, count=2)
    pass_ = encoder.begin_compute_pass(timestamp_writes={
        "query_set": query_set,
        "beginning_of_pass_write_index": 0,
        "end_of_pass_write_index": 1,
    })

See ``examples/compute_timestamps.py`` for the complete pattern.


RenderDoc
~~~~~~~~~~

`RenderDoc <https://renderdoc.org/>`_ can capture a frame, show all API calls,
inspect buffer contents, and display shader source.  Launch your Python script
under RenderDoc via *File → Launch Application*.

Common GPU Compute Bugs
~~~~~~~~~~~~~~~~~~~~~~~~

+-------------------------------------+-------------------------------------------+
| Bug                                 | Symptom / Cause                           |
+=====================================+===========================================+
| Missing ``workgroupBarrier()``      | Non-deterministic wrong results (race)    |
+-------------------------------------+-------------------------------------------+
| Wrong dispatch count                | Out-of-bounds buffer access (silent on    |
|                                     | GPU, crash on validation layers)          |
+-------------------------------------+-------------------------------------------+
| Buffer too small                    | wgpu raises a Python ``ValueError`` with  |
|                                     | a clear message about the binding         |
+-------------------------------------+-------------------------------------------+
| Forgetting ``COPY_SRC``             | ``read_buffer`` raises an error           |
+-------------------------------------+-------------------------------------------+
| Uniform buffer not 16-byte aligned  | Silently reads zeros / garbage            |
+-------------------------------------+-------------------------------------------+
| Wrong format in ``memoryview.cast`` | Wrong numbers in output                   |
+-------------------------------------+-------------------------------------------+

**Exercise 7:** Intentionally remove a ``workgroupBarrier()`` from
``examples/compute_tiled_matmul.py`` and add an assertion.  Run it several
times – does it fail consistently or only sometimes?


8. Advanced Topics
------------------

Indirect Dispatch
~~~~~~~~~~~~~~~~~~

Normally the workgroup count is a Python integer specified on the CPU.  With
*indirect dispatch* the counts live inside a GPU buffer, allowing the GPU itself
to decide how many workgroups to run (e.g. after a filtering pass):

.. code-block:: python

    import ctypes

    params = (ctypes.c_uint32 * 3)(nx, ny, nz)
    buf_indirect = device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.INDIRECT
    )

    pass_.dispatch_workgroups_indirect(buf_indirect, offset=0)

See ``tests/test_util_compute.py::test_compute_indirect`` for a tested example.


Multi-Pass Compute
~~~~~~~~~~~~~~~~~~~

Multiple compute passes can be chained in one ``CommandEncoder``.  Each pass
sees the output of the previous one because wgpu inserts the necessary memory
barriers between passes automatically:

.. code-block:: python

    encoder = device.create_command_encoder()

    # Pass 1: compute C = A @ B
    pass1 = encoder.begin_compute_pass()
    pass1.set_pipeline(matmul_pipeline)
    pass1.set_bind_group(0, matmul_bind_group)
    pass1.dispatch_workgroups(wg_x, wg_y, 1)
    pass1.end()

    # Pass 2: scale C in-place
    pass2 = encoder.begin_compute_pass()
    pass2.set_pipeline(scale_pipeline)
    pass2.set_bind_group(0, scale_bind_group)   # bind_group shares buf_c
    pass2.dispatch_workgroups(wg_x * wg_y, 1, 1)
    pass2.end()

    device.queue.submit([encoder.finish()])

Memory Barriers Between Passes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

wgpu inserts a full *pipeline barrier* (all-to-all memory dependency) between
consecutive passes in the same encoder.  This guarantees that writes from
pass *N* are visible to pass *N+1*.

Within a single pass, threads in *different* workgroups have no ordering
guarantee unless you use ``storageBarrier()`` (for storage-buffer writes visible
to subsequent dispatches in the same pass — a WebGPU 2 feature).

For cross-workgroup synchronisation in a single pass, prefer splitting into
multiple passes.


Async GPU Execution
~~~~~~~~~~~~~~~~~~~~

``device.queue.submit()`` is *non-blocking* – it returns immediately while the
GPU executes the commands asynchronously.  ``queue.read_buffer()`` implicitly
waits for completion.

For overlapping CPU and GPU work you can poll device completion or use the async
Python API (``await device.queue.on_submitted_work_done()``).


How wgpu Schedules Commands Internally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under the hood:

1. ``create_command_encoder()`` allocates a recording buffer.
2. Each ``begin_compute_pass()`` call emits a *pass descriptor* (timestamp
   queries, labels) into the recording.
3. ``dispatch_workgroups()`` records a *draw/dispatch call* with workgroup counts.
4. ``encoder.finish()`` validates the recording and seals it into a
   ``GPUCommandBuffer``.
5. ``queue.submit([cmd_buf])`` serialises the command buffer into the native
   backend's queue (``vkQueueSubmit`` on Vulkan, ``[commandBuffer commit]`` on
   Metal, etc.).
6. The GPU driver schedules execution across its SMs according to occupancy.

**Exercise 8:** Implement a two-pass compute pipeline that:

1. Pass 1: mark all values > threshold in an array (write 1 or 0 into a flag
   buffer).
2. Pass 2: use ``dispatch_workgroups_indirect`` where the dispatch counts are
   computed by pass 1 (advanced: requires an atomic counter in the first pass).
