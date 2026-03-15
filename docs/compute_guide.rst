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


9. GPU Performance Engineering
-------------------------------

This section teaches you to think like a GPU performance engineer.  Matrix
multiplication is used throughout as the concrete running example because it
exercises every optimization axis at once.

The complete, annotated example is in ``examples/compute_perf_matmul.py``.


9.1 Choosing Workgroup Sizes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The workgroup size (the argument to ``@workgroup_size`` in WGSL) is the single
most important tuning knob for a compute shader.

**Warp / wavefront alignment**

Hardware executes threads in *warps* (NVIDIA, 32 threads) or *wavefronts* (AMD,
64 threads).  If your workgroup size is not a multiple of the warp size, the
last warp contains idle lanes:

.. code-block:: text

    warp size = 32
    workgroup size = 40  →  warp 0: threads 0–31 (full)
                             warp 1: threads 32–39 ACTIVE, 40–63 IDLE (25% waste)

    workgroup size = 64  →  warp 0: threads 0–31 (full)
                             warp 1: threads 32–63 (full)  ← no waste

Rule of thumb: **always use a multiple of 64** (safe on both 32- and 64-thread
hardware).

**1-D problems (e.g. vector addition)**

Common good values: ``64``, ``128``, ``256``.  Larger workgroups increase the
number of threads the scheduler can use to hide memory latency, but beyond
~256 threads you rarely gain further benefit.

.. code-block:: wgsl

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) { ... }

**2-D problems (e.g. matrix multiplication)**

Use a square tile of side ``TILE``, giving ``TILE × TILE`` threads per workgroup.

.. code-block:: wgsl

    @compute @workgroup_size(16, 16, 1)   // 256 threads, TILE=16
    fn main(...) { ... }

Dispatch enough workgroups to cover the output matrix:

.. code-block:: python

    import math
    wg_x = math.ceil(N / TILE)
    wg_y = math.ceil(M / TILE)
    pass_.dispatch_workgroups(wg_x, wg_y, 1)

**Why TILE=16 is often optimal**

With TILE=16 each workgroup has 256 threads, which is:

* A multiple of 64 (warp-aligned on both AMD and NVIDIA).
* Small enough that many workgroups can be live simultaneously on each SM
  (high occupancy).
* Large enough to reuse each tile element 16 times before the next global
  memory fetch (good arithmetic intensity).


9.2 Memory Tiling
~~~~~~~~~~~~~~~~~~

Tiling is the technique of partitioning the problem into *tiles* that fit in
fast on-chip workgroup (shared) memory, loading each tile once from slow global
memory, and reusing it many times.

**Naive matmul – global memory bottleneck**

.. code-block:: wgsl

    // Each thread independently computes one element of C.
    // Thread (row, col) reads K elements from A's row and K from B's col.
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < K; k++) {
        acc += A[row * K + k] * B[k * N + col];
    }

Total global reads: ``2 × M × N × K`` floats.  For a 1024×1024 square matmul
that is ~2 billion reads, easily saturating global memory bandwidth.

**Tiled matmul – data reuse**

A workgroup of TILE×TILE threads cooperatively loads a TILE×TILE sub-block of
A and one of B into workgroup memory, then every thread in the workgroup reuses
those values TILE times before the next load:

.. code-block:: wgsl

    var<workgroup> tile_a: array<f32, TILE * TILE>;
    var<workgroup> tile_b: array<f32, TILE * TILE>;

    for (var t: u32 = 0u; t < num_tiles; t++) {
        // --- all TILE*TILE threads load one element each ---
        tile_a[lrow * TILE + lcol] = A[row * K + t * TILE + lcol];
        tile_b[lrow * TILE + lcol] = B[(t * TILE + lrow) * N + col];
        workgroupBarrier();

        // --- all threads compute TILE multiply-adds using on-chip data ---
        for (var k = 0u; k < TILE; k++) {
            acc += tile_a[lrow * TILE + k] * tile_b[k * TILE + lcol];
        }
        workgroupBarrier();
    }

Global reads with tiling: ``2 × M × N × K / TILE`` — a **TILE× reduction** in
global memory traffic.  For TILE=16 and a 1024×1024 matrix that is ~125 million
reads instead of 2 billion.

**The two-barrier pattern**

Two ``workgroupBarrier()`` calls are essential:

1. **After loading tiles** – ensures every element is written before any thread
   reads them (prevents read-before-write races).
2. **After accumulating** – ensures every thread finishes reading before any
   thread overwrites the tile for the next iteration (prevents
   write-after-read races).


9.3 Avoiding Bank Conflicts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Workgroup (shared) memory is implemented as an array of SRAM *banks*.  Most
modern GPUs have **32 banks**, each 4 bytes wide.  A float stored at byte
offset ``b`` lives in bank ``(b / 4) % 32``.

**What is a bank conflict?**

A bank conflict occurs when two or more threads in the same warp access
*different addresses* that map to the *same bank*.  The hardware must serialize
those accesses, adding one extra cycle per conflicting thread.

**Bank conflicts in a square tile**

With a tile of width TILE stored in row-major order (stride = TILE), the
address of element ``(row, col)`` is ``tile[row * TILE + col]``.

During the accumulation loop, thread ``(lrow, lcol)`` reads column ``k`` from
``tile_a``:

.. code-block:: text

    tile_a[lrow * TILE + k]

All threads with the same ``lrow`` (i.e., the same warp row) read the same
element — that is a *broadcast*, which hardware handles efficiently (no
conflict).

The problem is when threads in the *same warp column* read the same column
from ``tile_b``:

.. code-block:: text

    tile_b[k * TILE + lcol]

Here the row ``k`` varies across iterations, and the stride between rows is
exactly TILE floats.  When TILE is a multiple of the number of banks (32),
consecutive rows fall on the same bank:

.. code-block:: text

    TILE=32: row 0 → bank 0, row 1 → bank (32 % 32) = 0, row 2 → bank 0, …
    → 32-way bank conflict per warp!

**The padding fix (stride = TILE + 1)**

Declare the workgroup arrays with a *padded row stride* of ``TILE + 1``:

.. code-block:: wgsl

    const TILE_PAD: u32 = TILE + 1;
    var<workgroup> tile_a: array<f32, TILE * TILE_PAD>;
    var<workgroup> tile_b: array<f32, TILE * TILE_PAD>;

Access with the padded stride:

.. code-block:: wgsl

    tile_a[lrow * TILE_PAD + lcol]  // load
    tile_a[lrow * TILE_PAD + k]     // accumulate

Now row spacing = ``(TILE + 1) * 4`` bytes.  For TILE=16:
``17 * 4 = 68 bytes → bank 17 ≠ bank 0`` — successive rows land on
different banks.  For TILE=32: ``33 * 4 = 132 bytes → bank 1 ≠ bank 0``.

The padding wastes ``TILE × 4`` bytes per tile (one extra column of floats),
but the throughput improvement on real hardware is substantial for TILE=32.

See ``examples/compute_perf_matmul.py`` for the complete implementation with
inline analysis of which threads land on which banks.

**Verifying the fix**

Use a GPU profiler (NVIDIA Nsight Compute, AMD Radeon GPU Profiler) to inspect
the ``l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld`` counter.  After
applying padding it should drop to near zero.


9.4 Minimising Global Memory Access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every byte fetched from VRAM costs ~100–200 ns of latency and consumes precious
memory bandwidth.  The strategies below systematically reduce global memory
traffic.

**1. Exploit data reuse with tiling** (see §9.2)

The fundamental technique: move data from global → workgroup memory once,
then reuse it ``TILE`` times.

**2. Coalesced access – load consecutive addresses per warp**

All threads in the same warp should access a contiguous, aligned block of
global memory.  The hardware then issues a single wide read transaction
(128 bytes on most GPUs) instead of N separate ones.

Good (coalesced):

.. code-block:: wgsl

    // Thread lcol=0 reads col 0, lcol=1 reads col 1, …, lcol=15 reads col 15.
    // Addresses: row*K+t*TILE+0, row*K+t*TILE+1, …, row*K+t*TILE+15 → contiguous.
    tile_a[lrow * TILE_PAD + lcol] = mat_a[row * K + t * TILE + lcol];

Bad (strided):

.. code-block:: wgsl

    // If instead threads accessed different rows of A, addresses would be
    // spaced K floats apart → separate memory transactions per thread.
    tile_a[lrow * TILE_PAD + lcol] = mat_a[(row + lcol) * K + t * TILE];

**3. Use uniform / constant buffers for small read-only data**

Pipeline parameters (matrix dimensions, scaling factors) should live in a
``var<uniform>`` buffer, not a storage buffer.  The GPU caches uniform data
in a dedicated constant cache that is much faster than the L2 path used for
storage buffers.

.. code-block:: wgsl

    @group(0) @binding(3) var<uniform> dims: vec4<u32>;  // M, K, N, pad

**4. Read/write once per element**

The output ``mat_c`` is written exactly once per thread, at the very end of
the kernel after all tiles have been accumulated.  Avoid writing partial
results to global memory mid-kernel — that would add extra expensive stores.

**5. Arithmetic intensity**

*Arithmetic intensity* = FLOPs / bytes transferred.  Optimising a shader means
increasing this ratio.  For matrix multiplication:

.. code-block:: text

    FLOPs = 2 * M * N * K  (one multiply + one add per inner-product step)
    Bytes (naive)  = 2 * M * N * K * 4   → intensity = 0.5 FLOP/byte (DRAM-bound)
    Bytes (TILE=16) = 2 * M * N * K / 16 * 4 → intensity = 8 FLOP/byte (closer to compute-bound)

Hardware rooflines (representative values):

+--------------------+--------------------+-------------------+
| GPU class          | Peak FP32 (TFLOP/s)| Mem BW (TB/s)     |
+====================+====================+===================+
| Mid-range discrete | 10–20              | 0.3–0.6           |
+--------------------+--------------------+-------------------+
| High-end discrete  | 40–80              | 1–3               |
+--------------------+--------------------+-------------------+

The *ridge point* (the minimum arithmetic intensity required to be
compute-bound rather than memory-bound) is roughly ``Peak FP32 / Mem BW``,
typically **30–100 FLOP/byte** for modern discrete GPUs.

TILE=16 gives ~8 FLOP/byte — still below the ridge point, so larger tiles
(TILE=32) or register tiling can push towards compute-bound operation.


9.5 Occupancy Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Occupancy* is the fraction of the maximum number of resident warps that are
actually active on a streaming multiprocessor (SM) at any moment.

High occupancy is desirable because the GPU hides memory latency by switching
to another ready warp while the current warp waits for data.  With low
occupancy there are not enough warps to hide latency and the SMs stall.

**Resources that limit occupancy**

+---------------------------+--------------------------------------------------+
| Resource                  | Effect of excessive use                          |
+===========================+==================================================+
| Threads per SM            | Too many threads per workgroup → fewer           |
|                           | workgroups resident on the SM                   |
+---------------------------+--------------------------------------------------+
| Workgroup (shared) memory | Too many bytes per workgroup → fewer workgroups  |
|                           | fit in the SM's on-chip SRAM                    |
+---------------------------+--------------------------------------------------+
| Registers per thread      | Too many registers → fewer threads per SM        |
+---------------------------+--------------------------------------------------+

**Occupancy formula (simplified)**

For a shader with ``threads_per_wg`` threads and ``smem_per_wg`` bytes of
workgroup memory:

.. code-block:: text

    max_threads_per_SM  ≈ 2048  (typical modern GPU)
    max_smem_per_SM     ≈ 49152 bytes (48 KB)
    max_wg_per_SM_threads = max_threads_per_SM // threads_per_wg
    max_wg_per_SM_smem   = max_smem_per_SM    // smem_per_wg
    active_wg_per_SM     = min(max_wg_per_SM_threads, max_wg_per_SM_smem)
    occupancy            = active_wg_per_SM * threads_per_wg / max_threads_per_SM

**TILE=16 example**

.. code-block:: text

    threads_per_wg   = 16 * 16  = 256
    smem_per_wg      = 2 * 16 * 17 * 4  = 2176 bytes  (with TILE_PAD=17)

    max_wg (threads) = 2048 // 256  = 8
    max_wg (smem)    = 49152 // 2176 = 22
    active_wg        = min(8, 22)   = 8      ← thread-bound
    occupancy        = 8 * 256 / 2048 = 100%

**TILE=32 example**

.. code-block:: text

    threads_per_wg   = 32 * 32  = 1024
    smem_per_wg      = 2 * 32 * 33 * 4  = 8448 bytes  (with TILE_PAD=33)

    max_wg (threads) = 2048 // 1024 = 2
    max_wg (smem)    = 49152 // 8448 = 5
    active_wg        = min(2, 5)    = 2      ← thread-bound
    occupancy        = 2 * 1024 / 2048 = 100%

Both TILE=16 and TILE=32 reach 100% occupancy on this model GPU, but TILE=32
has only 2 active workgroups/SM compared to 8 for TILE=16.  With fewer
concurrent workgroups there is less opportunity to hide the latency of the
``workgroupBarrier()`` stalls, so TILE=16 often performs better in practice.

**Occupancy vs arithmetic intensity trade-off**

Increasing TILE improves arithmetic intensity (fewer global reads per FLOP)
but reduces concurrency (fewer workgroups per SM).  The optimal tile size
depends on the hardware's memory latency and bandwidth.  Always measure:

.. code-block:: python

    # Use compute_timestamps.py as a template:
    device = adapter.request_device_sync(
        required_features=[wgpu.FeatureName.timestamp_query]
    )
    # ... setup query_set and pass timestamp_writes to begin_compute_pass ...

**Practical recommendations**

1. Start with TILE=16 (256 threads, a safe default on all modern GPUs).
2. Profile with ``compute_timestamps.py`` at TILE=8, 16, 32.
3. Prefer smaller tiles when the matrix K dimension is small (fewer
   reuse opportunities mean tiling overhead dominates).
4. Use a GPU profiler to check occupancy and memory efficiency counters
   rather than relying solely on analytic estimates.

**Exercise 9:** Open ``examples/compute_perf_matmul.py``, run it with TILE=8,
16, and 32, and compare the printed performance summaries.  Then add GPU
timestamp queries (see ``examples/compute_timestamps.py``) to measure actual
kernel execution time and compute the achieved GFLOP/s for each tile size.
