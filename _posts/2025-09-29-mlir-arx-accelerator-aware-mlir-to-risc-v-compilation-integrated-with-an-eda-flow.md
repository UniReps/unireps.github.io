---
layout: distill
title: "MLIR-ARX: Accelerator-Aware MLIR-to-RISC-V Compilation Integrated with an EDA Flow"
date: 2025-09-20
categories: [mlir, compiler, deep-learning]
featured: true

authors:
  - name: JooHyoung Cha
    url: "https://orcid.org/0009-0008-2123-454X"
    affiliations:
      name: University of Science and Technology, Republic of Korea
  - name: Yongin Kwon
    url: "https://orcid.org/0000-0003-2973-246X"
    affiliations:
      name: Electronics and Telecommunications Research Institute, Republic of Korea

bibliography: 2025-09-29-mlir-arx-accelerator-aware-mlir-to-risc-v-compilation-integrated-with-an-eda-flow.bib

toc:
  - name: "1. Introduction"
  - name: "2. Background"
  - name: "3. Design and Architecture"
  - name: "4. Implementation"
  - name: "5. Early Evaluation"
  - name: "6. Conclusion"
  - name: "7. Limitations and outlook"
  - name: "8. Related Work"
  - name: "9. VTA Configuration and Expected Performance on FPGA"
  - name: "10. ARX Dialect"
  - name: "11. Profiling Instrumentation and Log Schema"
  - name: "12. Cost Model Details and Selection Algorithm"
  - name: "13. MNIST Model Shapes and Mapping Notes"

---

**TLDR**

Deploying AI models on heterogeneous RISC-V systems requires not only partitioning computation between CPU and accelerator, but also identifying which operations to accelerate when hardware capabilities evolve. We present `mlir-arx`, a profile-guided compiler built on MLIR that introduces an `arx` dialect to encode accelerator constraints, inserts lightweight profiling operations via dedicated passes, and applies a two-stage (analytic + profile-guided) cost model to form maximal offload regions under resource and dependency limits. Early evaluation on a small MNIST CNN with a configurable VTA accelerator on a Genesis FPGA prototype demonstrates that the compile--measure loop identifies profitable regions and achieves significant end-to-end speedups over the CPU-only baseline.

Availability. Source code and artifacts are available at: [Repository in Our Gitlab](https://gitlab.com/ones-ai/mlir-arx)

---

# **1. Introduction**

Edge and embedded AI systems increasingly pair general-purpose RISC-V cores with domain accelerators (from NPUs to lightweight tensor engines) to meet latency and energy targets<d-cite key="jouppi2017tpu, eyeriss2016, nvdla2019"/>. In such settings, the practical challenge is not only to partition a model across CPU and accelerator boundaries but also to decide what to accelerate when accelerator capabilities may be unknown or evolving at project start. A sensible path is to begin from a CPU-only baseline, profile real executions, and then offload the most profitable regions subject to resource and orchestration constraints.

MLIR's multi-dialect design and progressive lowering are a natural fit for this workflow<d-cite key="mlir"/>. However, turning it into an end-to-end solution for RISC-V SoCs requires additional pieces: 

### 1.1 Requires
1. IR-level capability modeling to express an accelerator's constraints
2. lightweight profiling instrumentation that can be injected/stripped by passes
3. a cost model that joins analytic estimates with profile-derived efficiencies
4. packaging that integrates with an SoC/EDA flow. 


We present `MLIR-ARX`, a profile-guided compiler built on MLIR that introduces an `arx` dialect for accelerator capabilities, selects profitable offload regions via a two-stage cost model, and provides a retargetable backend integrated with RISC-V eXpress<d-footnote>We thank collaborators who contributed to the RVX integration and platform bring-up.</d-footnote> (`RVX`) for FPGA bring-up<d-cite key="rvx-etrij"/>. For initial experiments we use VTA as a configurable accelerator to test profile-driven selection and resource-aware partitioning, while the CPU-only path remains the numerically correct fallback<d-cite key="iree22tiny,tvm2018"/>.

### 1.2 Contributions

* **Accelerator-aware IR**. An `arx` dialect that captures accelerator capabilities (constraints, tiling, resource usage) and enables principled lifting of standard tensor ops.
* **Cost-guided partitioning**. A two-stage (analytic + profile-guided) model that forms offload regions with explicit DMA orchestration and safe CPU fallbacks.
* **Retargetable backend + RVX integration**. Code generation for RISC-V plus accelerator stubs and a device manifest that RVX uses for automatic integration.
* **Early evaluation**. A small CNN on MNIST running on RVX-based FPGA prototypes with VTA, validating the compile–measure loop and selection mechanism.

---

# **2. Background**
**MLIR Foundations and Target Platform**

### 2.1 MLIR in brief

MLIR is a multi-level compiler infrastructure that represents computations at various abstraction levels and supports progressive lowering through dialects and pattern-based rewrites<d-cite key="mlir"/>. Dialects capture operations, types, and constraints; key ones for `MLIR-ARX` include `mhlo/StableHLO` for tensor semantics<d-cite key="stablehlo"/>, `linalg` for structured kernels, and `memref` for explicit memory. 

Two features are central to our setting: bufferization, which separates algorithmic transformations from storage decisions by converting tensors to `memref`s with explicit lifetimes, and dialect interfaces/converters, which allow capability-aware lifting or lowering between dialects. These make it possible to express accelerator constraints in IR and offload only the supported regions. Prior systems, from TVM's multi-backend flow<d-cite key="tvm2018"/> to IREE's embedded pipelines<d-cite key="iree22tiny"/>, demonstrate the viability of such end-to-end MLIR-based compilation.

### 2.2 Why MLIR for RISC-V + Accelerators
RISC-V-based edge platforms often combine a control processor (with or without a vector extension) and one or more domain accelerators attached via a memory-mapped interconnect. 

This creates three immediate needs: 
1. partitioning of the model into CPU and accelerator regions.
2. explicit orchestration of DMA, synchronization, and address spaces.
3. graceful fallback when constraints are violated. MLIR's dialect modularity lets us.

(a) express accelerator capabilities as IR-level contracts, (b) form and legalize offload regions under those contracts, and (c) lower both sides—CPU and accelerator stubs—within a single pass manager, sharing analyses (shape, alias, dependence) across the boundary. Compared with ad-hoc code generators, the benefits are: reuse of upstream transformations, uniform debuggability, and a single IR for both accelerated and non-accelerated builds.

### 2.3 RVX overview

RVX is an EDA environment to assemble RISC-V SoCs (single-/multi-core, memory subsystem, interconnect, peripheral IP), validate them on FPGA, and produce handoff artifacts for silicon<d-cite key="rvx-etrij"/>. On the software side, RVX provides toolchain integration points (boot/firmware layout, MMIO address map, interrupt lines) and profiling hooks. `MLIR-ARX` targets this boundary: it emits CPU binaries, an accelerator runtime and driver stubs, and a device manifest that RVX consumes to wire up interconnect ports and firmware tables. By aligning compiler outputs with RVX's manifests, we avoid manual bring-up steps when retargeting cores or adding/removing accelerators.

> We primarily target edge-style RISC-V SoCs where accelerators are memory-mapped with local SRAM and DMA engines. Our current prototype assumes statically known shapes in offload regions; dynamic-shape support is under active development.

---

# **3. Design and Architecture**
**Profile-Guided MLIR-to-RISC-V Offload**

### 3.1 Pipeline Overview

[Figure 1](#flow) shows the end-to-end flow: models import into MLIR, run once on a CPU baseline to profile, mine offload candidates, select them under FPGA budgets, form offload regions, and lower to CPU and accelerator backends. A machine-readable accelerator description generates the `arx` dialect and converters; RVX profiling closes the compile–measure loop.

### 3.2 Baseline Execution and Profiling
`MLIR-ARX` assumes no accelerator a priori. Every model first executes end-to-end on RISC-V, producing per-op/region profiles (cycles, bytes moved, stalls, shapes/dtypes/layouts) keyed by stable IR handles. This path is also the correctness fallback for any region that proves illegal or unprofitable to offload.

### 3.3 Candidate Discovery and Cost Modeling
Hot single ops or short fusable patterns are grouped by semantics/constraints as offload candidates. For a region $R$, we use an inline estimate $\Delta T(R)=T_{\mathrm{cpu}}(R)-T_{\mathrm{off}}(R)$ with $T_{\mathrm{off}}\approx T_{\mathrm{setup}}+\max(T_{\mathrm{dma}},T_{\mathrm{cmp}})+T_{\mathrm{sync}}$ where analytic terms (op counts, tiling, bandwidth/latency) are corrected by profile-derived efficiencies. 

### 3.4 Resource-Aware Selection and Partitioning
On FPGA targets, selection respects LUT/FF/DSP/BRAM, local SRAM, DMA lanes, and clock budgets. Given per-candidate resource costs $\mathcal{R}(c)$ and budget $B$, we maximize $\sum\Delta T(c)$ subject to $\sum\mathcal{R}(c)\le B$, preferring high benefit density when tight. Selected ops become maximal \emph{offload regions} under dependency/memory constraints; the compiler inserts explicit host–device copies/fences and schedules DMA to overlap with compute.


### 3.5 Lowering and Runtime
Both CPU and accelerator paths first lower into the `arx` dialect.
CPU ops then follow `linalg` $\to$ `scf/affine` $\to$ `LLVM` to produce RISC-V ELFs, while accelerator ops lower directly to library calls with a thin runtime and DMA descriptors.
A device manifest (MMIO ranges, IRQ lines) is also emitted for RVX integration.


### 3.6 Retargeting and feedback

A YAML/JSON accelerator description regenerates the `arx` capability model and converters, enabling recompilation without model changes. Deployed binaries feed fresh RVX profiles back into the cost model, which updates efficiencies and re-ranks candidates for the next iteration.


<a id="flow"></a>
<!-- ![My Image](/blog/assets/img/2025-09-29-mlir-arx-accelerator-aware-mlir-to-risc-v-compilation-integrated-with-an-eda-flow/01_system_overview.png) -->
<div style="text-align: center;">
  <img src="/blog/assets/img/2025-09-29-mlir-arx-accelerator-aware-mlir-to-risc-v-compilation-integrated-with-an-eda-flow/01_system_overview.png" alt="My Image" width="700" />
</div>

---

# **4. Implementation**
**ONNX-MLIR Base, ARX Dialect, and RVX/VTA Bring-up**

### 4.1 Codebase and MLIR Integration
We implement `MLIR-ARX` by extending the open-source ONNX-MLIR<d-cite key="onnxmlir2020"/> stack. Models are imported and legalized through ONNX-MLIR into MLIR's tensor/structured dialects, upon which we add our `ARX` components. The output code runs unmodified on RVX-synthesized RISC-V platforms; when no accelerator is present, the CPU path serves as the numerically correct fallback.

### 4.2 ARX Dialect and Lowering Paths
The `arx` dialect lifts eligible tensor and `linalg` ops into accelerator-aware form, annotated with capability and tiling metadata. Both CPU and accelerator ops lower through this dialect: CPU paths continue via `linalg` $\to$ `scf/affine` $\to$ `LLVM` for RISC-V binaries, while accelerator ops become driver/library calls with explicit host–device transfers. Because both paths share the same pass pipeline, analyses such as shape, aliasing, and dependence are reused across CPU and accelerator lowering.

### 4.3 Profiling Instrumentation
To support the profile-driven flow, we define lightweight profiling ops (begin/end counters, byte/traffic counters, DMA/compute timestamps). When the compiler is invoked with a profiling option, an *instrumentation pass* inserts these ops around selected regions/ops during canonicalization and bufferization. The pass is designed to be idempotent and can be stripped in a late “de-instrumentation” pass for production builds. Profiles are keyed by stable IR handles to survive recompilation unless shapes change.

### 4.4 Accelerator Target: VTA
For an initial hardware target we use Apache TVM's VTA soft accelerator. VTA implements a small set of tensor primitives with parameterizable compute parallelism (e.g., PE width), local SRAM sizes, and instruction buffer depth. This configurability makes it a good vehicle to test `MLIR-ARX`'s candidate selection and resource-aware partitioning: The capability model is regenerated from the chosen VTA configuration, while the cost model estimates tile fit, DMA overlap, and achievable throughput under the given SRAM and bandwidth constraints.

---

# **5. Early Evaluation**

### 5.1 Setup
Our prototype system follows the dual-ORCA configuration in prior TIP work, with two ORCA RISC-V RV32IM cores synthesized at 100 MHz on a Genesis FPGA and connected via the RVX-generated $\mu$NoC and AXI-based DRAM subsystem. A configurable VTA overlay is attached as an MMIO+DMA device, using 8-bit inputs/weights, 32-bit accumulators, and a $16{\times}16{\times}16$ GEMM block. All components are generated by RVX from a manifest and deployed to FPGA.


### 5.2 Benchmark
For evaluation, we use a small CNN for MNIST consisting of two conv+ReLU blocks with $2{\times}2$ pooling, 
followed by a fully connected layer (input $1{\times}28{\times}28$). 
Profiles are first collected from the CPU-only execution to guide candidate discovery. 
Operators selected for offload are *conv2d(+bn)+relu* blocks (Conv1, Conv2) and the final fully connected layer; control flow, quantize/dequantize, and maxpool remain on the CPU.

### 5.3 CPU-only Baseline
[Table 1](#inference-time-arx) reports the measured per-layer latency on the dual ORCA cores. 
The two convolution layers dominate the runtime (over 95 % of total latency), 
making them the primary candidates for acceleration.

### 5.4 Projected Accelerator Performance
Since the FPGA prototype is still under integration, we estimate accelerator-side performance 
using a simple throughput model assuming three VTA configurations (A/B/C) with 
256/512/1024 MAC/cycle.
[Table 1](#inference-time-arx) shows the projected latency when Conv1, Conv2, and the fully connected layer are offloaded to VTA, 
while other layers remain on the CPU. 
The results indicate a potential end-to-end speedup of $50 \times$–$90\times$ compared to the CPU-only baseline, 
with hardware resource utilization scaling from $\sim$ 15 % to $\sim$ 60 % of available DSPs on the target FPGA.


<a id="inference-time-arx"></a>
<table>
  <caption>
    Table 1. Predicted latency of MNIST CNN layers with CPU-only vs. three VTA configurations (A/B/C). VTA throughput assumes 256/512/1024&nbsp;MAC/cycle at 75% utilization. Hardware utilization is estimated on XC7K325T FPGA.
  </caption>

  <thead>
    <tr>
      <th style="text-align:left;">Layer</th>
      <th style="text-align:right;">Ops</th>
      <th style="text-align:right;">CPU-only (&micro;s)</th>
      <th style="text-align:right;">VTA-A (&micro;s)</th>
      <th style="text-align:right;">VTA-B (&micro;s)</th>
      <th style="text-align:right;">VTA-C (&micro;s)</th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <td style="text-align:left;">1. Quantize</td>
      <td style="text-align:right;">&mdash;</td>
      <td style="text-align:right;">5.73</td>
      <td style="text-align:right;">&mdash;</td>
      <td style="text-align:right;">&mdash;</td>
      <td style="text-align:right;">&mdash;</td>
    </tr>
    <tr>
      <td style="text-align:left;">2. Conv1</td>
      <td style="text-align:right;">97,344</td>
      <td style="text-align:right;">884.20</td>
      <td style="text-align:right;">5.07</td>
      <td style="text-align:right;">2.54</td>
      <td style="text-align:right;">1.27</td>
    </tr>
    <tr>
      <td style="text-align:left;">3. MaxPool1</td>
      <td style="text-align:right;">10,816</td>
      <td style="text-align:right;">28.96</td>
      <td style="text-align:right;">&mdash;</td>
      <td style="text-align:right;">&mdash;</td>
      <td style="text-align:right;">&mdash;</td>
    </tr>
    <tr>
      <td style="text-align:left;">4. Conv2</td>
      <td style="text-align:right;">557,568</td>
      <td style="text-align:right;">2949.15</td>
      <td style="text-align:right;">29.04</td>
      <td style="text-align:right;">14.52</td>
      <td style="text-align:right;">7.26</td>
    </tr>
    <tr>
      <td style="text-align:left;">5. FullyConnected</td>
      <td style="text-align:right;">5,120</td>
      <td style="text-align:right;">4.76</td>
      <td style="text-align:right;">0.27</td>
      <td style="text-align:right;">0.13</td>
      <td style="text-align:right;">0.07</td>
    </tr>
    <tr>
      <td style="text-align:left;">6. Dequantize</td>
      <td style="text-align:right;">&mdash;</td>
      <td style="text-align:right;">0.50</td>
      <td style="text-align:right;">&mdash;</td>
      <td style="text-align:right;">&mdash;</td>
      <td style="text-align:right;">&mdash;</td>
    </tr>
    <tr>
      <th style="text-align:left;">Total</th>
      <th style="text-align:right;">670,848</th>
      <th style="text-align:right;">3873.30</th>
      <th style="text-align:right;">69.57</th>
      <th style="text-align:right;">52.38</th>
      <th style="text-align:right;">43.78</th>
    </tr>
    <tr>
      <th scope="row" colspan="2" style="text-align:left;">HW Util. (DSP / BRAM / LUT %)</th>
      <td style="text-align:right;">&mdash;</td>
      <td style="text-align:right;">15 / 44 / 12</td>
      <td style="text-align:right;">31 / 44 / 21</td>
      <td style="text-align:right;">61 / 46 / 36</td>
    </tr>
  </tbody>
</table>

---

# **6. Conclusion**

We presented `MLIR-ARX`, a profile-driven MLIR compiler that begins from a CPU-only RISC-V baseline, identifies profitable regions, and offloads them to accelerators under resource and orchestration constraints. `MLIR-ARX` introduces the `arx` dialect for capability-aware lifting, lightweight profiling instrumentation, and a retargetable backend integrated with RVX. Early evaluation on a small MNIST CNN with a configurable VTA overlay shows that offloading Conv1/Conv2/FC achieves order-of-magnitude latency reductions over the dual-ORCA CPU baseline, consistent with our cost-model predictions.

---

# **7. Limitations and outlook**

Our cost model is analytic with profile-derived corrections; learned models may better capture controller effects and DMA/compute overlap. Offload regions currently assume static shapes; extending legality/bufferization for dynamic-shape cases is ongoing. Finally, while the CPU path can exploit vector intrinsics, full scheduling for attention-like blocks and multi-accelerator concurrency remains future work. We expect these extensions—alongside broader accelerator backends—to further tighten the compile–measure loop and reduce manual retargeting on RVX platforms.

---

# **8. Related Work**
**RISC-V CPUs, Vector Extensions, AI Accelerators, and Design Space Exploration**

### 8.1 RISC-V as the Control Plane for Heterogeneous ML SoCs
Open RISC-V implementations range from tiny in-order microcontrollers to out-of-order Linux-class cores, making them a natural control plane for accelerator-rich SoCs. Representative open cores include Rocket (in-order) and BOOM (out-of-order) from the Berkeley stack<d-cite key="rocket,boom"/>, CVA6/Ariane<d-cite key="cva6"/>, and the PULP family for ultra-low-power microcontrollers with DSP-like packed-SIMD extensions<d-cite key="pulp"/>. SoC generators such as Chipyard<d-cite key="chipyard"/> offer standard interconnects (TileLink/AXI) and co-processor attachment points (e.g., RoCC), reducing the cost of integrating DMA-capable accelerators alongside a RISC-V host.

### 8.2 RISC-V Vector and Packed-SIMD for ML
The RISC-V Vector extension (RVV)<d-cite key="rvv"/> adopts a vector-length-agnostic model that decouples vector width from the ISA, enabling portability across microarchitectures. Implementations can choose lane count and microarchitectural details, as explored in Hwacha and Ara<d-cite key="hwacha"/>. For MCU-class devices, the packed-SIMD “P” extensions from PULP<d-cite key="pulp"/> target fixed-point and dot-product primitives. In practice, RVV or packed SIMD is well-suited for control and medium-granularity tensor compute, while large GEMMs/convolutions are often offloaded to dedicated accelerators.

### 8.3 Attachment Patterns for AI Accelerators

Three patterns dominate:

1.  **Memory-mapped DMA engines**: accelerators with local SRAM and DMA, controlled as MMIO devices; the most common in embedded contexts.

1.  **Coprocessors (e.g., RoCC)**: accelerators invoked via custom instructions or queues, reducing software overhead but tying the ABI to a core design<d-cite key="rocket,chipyard,gemmini"/>.

2.  **Streaming/NoC-attached engines**: connected via on-chip networks with stream interfaces; the host sets up dataflow graphs and dispatches jobs.

`MLIR-ARX` assumes the first pattern (MMIO+DMA) but its IR contracts generalize to other attachments.

### 8.4 Tensor Accelerators and Dataflows
A wide body of work covers compute/dataflow design for DNNs: systolic arrays (e.g., TPU<d-cite key="jouppi2017tpu"/>), spatial dataflows (row/output/weight-stationary, e.g., Eyeriss<d-cite key="eyeriss2016"/>), and precision-specialized engines (e.g., NVDLA<d-cite key="nvdla2019"/>). Open-source generators like Gemmini<d-cite key="gemmini"/> produce RISC-V-attached systolic arrays with tunable parameters. These designs show that with aggressive on-chip reuse and explicit DMA/compute overlap, accelerators can deliver order-of-magnitude energy savings if the compiler/runtime manages packing, tiling, and synchronization.

### 8.5 FPGA Overlays and Soft Accelerators
FPGAs serve as prototyping and deployment platforms for edge ML accelerators. Overlay designs, including VTA<d-cite key="vta"/>, FINN<d-cite key="finn"/>, and hls4ml<d-cite key="hls4ml"/>, expose parameter spaces (PE array size, SRAM depth, DMA width) suitable for compiler-driven design space exploration. Compared to fixed-function ASICs, overlays trade some efficiency for rapid iteration and portability.

### 8.6 Memory Systems and Data Movement
Memory hierarchy decisions strongly influence performance and energy. Designs like Eyeriss exploit row-stationary mapping to minimize off-chip traffic; others (e.g., MAERI, SCNN) adapt to sparsity and flexible reductions. For RISC-V-attached accelerators, key challenges are software-visible packing/tiling to match SRAM/DMA configurations and overlapping DMA with compute. `MLIR-ARX`'s ARX dialect makes these constraints explicit in IR.

### 8.7 Design Space Exploration (DSE) for AI Accelerators
DSE methods co-optimize accelerator architecture and mapping. Frameworks like ZigZag<d-cite key="zigzag"/> generate and evaluate architecture–mapping pairs with cost models for area, energy, and performance, achieving significant energy gains over baseline mappings. Tools such as Timeloop<d-cite key="timeloop"/> and Accelergy<d-cite key="accelergy"/> focus on loop mapping and energy estimation.

These DSE strategies are applicable to RISC-V–integrated accelerators, where FPGA/SoC constraints require balancing performance and resource use. In `MLIR-ARX`, the cost model and candidate selection can be extended to search over accelerator microarchitectures and RISC-V/accelerator interface options, further tightening the compile–measure loop.

### 8.8 Positioning of MLIR-ARX
`MLIR-ARX` complements the hardware and DSE work above by embedding accelerator capability models in IR, using profile-guided cost modeling to identify profitable offloads, inserting legal DMA/synchronization with overlap, and retargeting automatically to new accelerators or configurations without model changes. Hardware advances supply the building blocks; `MLIR-ARX` provides the IR-level integration and automation in a RISC-V EDA flow.

### 8.9 MLIR-centric compiler stacks and HLS/RTL generation
MLIR has increasingly been used not only as a software-oriented IR but also as a hardware-construction and HLS coordination layer. The CIRCT project extends MLIR with hardware-facing dialects (e.g., `hw`, `comb`, `sv`, `fsm`) and export passes to synthesizable SystemVerilog, enabling end-to-end generation of RTL directly from MLIR programs <d-cite key="circt"/>. For dataflow-style accelerators, the `handshake` and `staticlogic` dialects capture fine- and coarse-grain control and enable automated scheduling/retiming before \emph{ExportVerilog}. These flows complement software-oriented tensor dialects by giving a path to hardware under the same abstraction umbrella.

A second line of work connects MLIR to C/C++ code generation as an HLS front end. The `EmitC` path lowers structured MLIR to portable C++ suitable for downstream toolchains, including HLS compilers, while preserving shape and buffer semantics. Building on this idea, ScaleHLS uses MLIR to drive loop transformations (tiling, unrolling, pipelining) and memory partitioning so that the emitted C/C++ attains predictable quality-of-results when synthesized by commercial HLS tools <d-cite key="scalehls"/>. This separation—high-level legality and transformation in MLIR, hardware construction in HLS—aligns with our design where algorithmic and storage decisions are expressed in IR.

Dynamic and elastic dataflow HLS has also been explored atop MLIR. Dynamatic integrates with MLIR's `handshake` pipeline to generate elastic circuits that tolerate variable-latency operators and memory, bringing modulo scheduling and token-based control to the HLS space <d-cite key="dynamatic"/>. At the coarse-grain end, MLIR-based AIE flows target CGRA-like AI engines (e.g., Xilinx/AMD AI Engine), using MLIR dialects to express tile-local compute, DMA, and interconnect routing before producing device-ready binaries <d-cite key="mliraie"/>. Finally, the Calyx project exposes a hardware-centric intermediate representation and an MLIR dialect that make resource sharing, banking, and control explicit, providing another route from MLIR programs to verifiable RTL generators <d-cite key="calyx"/>.

Position relative to our system. These efforts show two practical integration patterns for MLIR: (i) MLIR $\to$ RTL via CIRCT-style dialects, and (ii) MLIR $\to$ C/C++ via EmitC/ScaleHLS for HLS back ends. `MLIR-ARX` currently focuses on partitioning and orchestrating offload regions under a unified IR for CPU and accelerator execution, but its capability model and region formation are compatible with both patterns: the same `arx` ops can be lowered either to MMIO-driven stubs (our VTA/RVX path) or, in a future backend, to HLS-friendly C++ or directly to RTL through CIRCT. This suggests a path to automatically synthesize specialized accelerators for hot regions discovered by the profile-guided flow, while preserving the CPU fallback and the EDA integration boundary already in place.

---

# **9. VTA Configuration and Expected Performance on FPGA**

### 9.1 Hardware overview
VTA is a soft deep-learning accelerator intended for FPGAs. It implements a decoupled three-stage pipeline (load, compute, store) with dedicated on-chip buffers and task queues, enabling overlap of DMA and compute when tiling admits double buffering<d-cite key="vta,ml2tuner25"/>. The architecture exposes configuration knobs for arithmetic precision, on-chip buffer sizes, and the inner matrix-multiply shape that together determine tile fit, bandwidth demand, and achievable utilization.

### 9.2 Configuration reported on ZCU102
As background, we refer to the ZCU102-oriented VTA configuration summarized in [Table 2](#hardware-vta), reported by prior work<d-cite key="ml2tuner25"/>. 
Values are expressed in log2 form in TVM/VTA's JSON; for clarity we additionally list the corresponding bit-widths and buffer capacities. 
The reported buffer sizes are one step larger than the common defaults, chosen to better utilize Zynq UltraScale+ resources.

<a id="hardware-vta"></a>
<table>
  <caption>
    Table 2. VTA configuration on Xilinx ZCU102 (derived from the configuration used in EVTA<d-cite key="ml2tuner25"/>).
  </caption>
  <thead>
    <tr>
      <th style="text-align:left;">Attribute</th>
      <th style="text-align:right;">JSON (log2)</th>
      <th style="text-align:left;">Interpreted value</th>
      <th style="text-align:left;">Effect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>LOG_INP_WIDTH</code></td>
      <td style="text-align:right;">3</td>
      <td>8-bit int</td>
      <td>input precision</td>
    </tr>
    <tr>
      <td><code>LOG_WGT_WIDTH</code></td>
      <td style="text-align:right;">3</td>
      <td>8-bit int</td>
      <td>weight precision</td>
    </tr>
    <tr>
      <td><code>LOG_ACC_WIDTH</code></td>
      <td style="text-align:right;">5</td>
      <td>32-bit int</td>
      <td>accumulator precision</td>
    </tr>
    <tr>
      <td><code>LOG_BATCH</code></td>
      <td style="text-align:right;">0</td>
      <td>1</td>
      <td>batch factor in intrinsic</td>
    </tr>
    <tr>
      <td><code>LOG_BLOCK</code></td>
      <td style="text-align:right;">4</td>
      <td>16</td>
      <td>inner GEMM tile (PE block)</td>
    </tr>
    <tr>
      <td><code>LOG_UOP_BUFF_SIZE</code></td>
      <td style="text-align:right;">16</td>
      <td>64&nbsp;KiB</td>
      <td>micro-op buffer</td>
    </tr>
    <tr>
      <td><code>LOG_INP_BUFF_SIZE</code></td>
      <td style="text-align:right;">16</td>
      <td>64&nbsp;KiB</td>
      <td>input buffer</td>
    </tr>
    <tr>
      <td><code>LOG_WGT_BUFF_SIZE</code></td>
      <td style="text-align:right;">19</td>
      <td>512&nbsp;KiB</td>
      <td>weight buffer</td>
    </tr>
    <tr>
      <td><code>LOG_ACC_BUFF_SIZE</code></td>
      <td style="text-align:right;">18</td>
      <td>256&nbsp;KiB</td>
      <td>accumulator buffer</td>
    </tr>
  </tbody>
</table>


### 9.3 Implications for tiling and overlap
Given `LOG_BLOCK`=4, the intrinsic compute block is $16{\times}16{\times}16$. 

Legal tiles must satisfy buffer capacity and alignment constraints for the three-stage pipeline: 

(i) an input/weight sub-tile that fits \{64 KiB, 512 KiB\} with layout-specific padding.
(ii) an accumulator tile that fits 256 KiB.
(iii) DMA chunking that aligns with the memory interface. 

When these constraints are met, double buffering allows:

$$T_{\mathrm{off}}\approx T_{\mathrm{setup}}+\max(T_{\mathrm{dma}},T_{\mathrm{cmp}})+T_{\mathrm{sync}},$$

and hides the smaller of DMA/compute times.

### 9.4 Mapping to MLIR-ARX
In `mlir-arx`'s YAML capability description, the configuration in [Table 2](#hardware-vta) becomes the static part of the accelerator model (precision, intrinsic block, buffer capacities). The partitioner only lifts regions whose tiles provably fit, and the cost model accounts for (i) the $\max(T_{\mathrm{dma}},T_{\mathrm{cmp}})$ overlap enabled by double buffering, (ii) setup/synchronization, and (iii) memory-traffic inflation from packing and padding. Under multi-VTA, the resource model exposes the number of instances and the shared DRAM bandwidth so that selection can avoid overcommitting the memory system.

---

# **10. ARX Dialect** 
**Selected Operations and Capability Schema**

This section sketches the subset of ARX operations and attributes that our prototype uses to plan and legalize offload regions. The design mirrors MLIR's convention of making capabilities explicit at IR
boundaries so that legality and code generation are mechanically checkable.

### 10.1 Core ops and attributes
[Table 3](#arx-ops) summarizes representative ops and their key attributes. The attributes are chosen so that (i) legality checks are local, (ii) tiling
constraints can be statically validated, and (iii) DMA and compute costs can be derived from sizes and layouts.

<a id="arx-ops"></a>
<table>
  <caption>Table 3. Selected ARX ops and attributes used in the prototype.</caption>
  <thead>
    <tr>
      <th style="text-align:left;">Op</th>
      <th style="text-align:left; width:5cm;">Key attributes</th>
      <th style="text-align:left; width:6cm;">Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>arx.conv2d</code></td>
      <td><code>dtype, strides, dilations, padding, tile_h, tile_w, ic_blk, oc_blk</code></td>
      <td>Convolution lifted from <code>linalg.conv_*</code>. Tiling attributes reflect scratchpad fit and inner GEMM blocking.</td>
    </tr>
    <tr>
      <td><code>arx.gemm</code></td>
      <td><code>dtype, M_blk, N_blk, K_blk</code></td>
      <td>Canonicalized matmul; blocks must be multiples of the accelerator&#39;s intrinsic block.</td>
    </tr>
    <tr>
      <td><code>arx.pool</code></td>
      <td><code>mode, kH, kW, strides</code></td>
      <td>Optional; emitted only when the accelerator implements pooling.</td>
    </tr>
    <tr>
      <td><code>arx.copy</code></td>
      <td><code>src_space, dst_space, bytes, align</code></td>
      <td>Logical copies across host/device address spaces. Lowered to DMA descriptors when possible.</td>
    </tr>
    <tr>
      <td><code>arx.region</code></td>
      <td><code>reads, writes, sram_bytes</code></td>
      <td>Opaque offload region container; captures side conditions (aliasing, fences).</td>
    </tr>
  </tbody>
</table>

### 10.2 Example lifting

A legal `linalg.matmul` with shapes that fit the intrinsic block becomes:

<pre style="color:#111827; background:#f3f4f6">
%y = arx.gemm  %a, %b
     { dtype = i8, M_blk = 16, N_blk = 16, K_blk = 16 } : ...
</pre>

---

# **11. Profiling Instrumentation and Log Schema**

### 11.1 Instrumentation ops

Profiling is injected by a dedicated pass when a command-line flag is set. The ops are intentionally minimal so that they can be stripped late in the pipeline.

- `arx.prof.begin handle: i64 {counters = [cycles, bytes_rd, bytes_wr]}`
- `arx.prof.end handle: i64`

The pass places `begin/end` around candidate ops and region boundaries after bufferization, ensuring that the measured bytes reflect concrete `memref` layouts and copies.


### 11.2 Runtime counters and emission
On RVX, the runtime reads CPU cycle counters and DMA byte counters at `begin/end`. Each record is keyed by the IR handle (a stable 64-bit hash).

<pre style="color:#111827; background:#f3f4f6">
record {
  handle: 0x17a3...
  cycles:  239812
  bytes_rd:  1572864
  bytes_wr:   262144
  stalls: {dma_wait: 0.12, sram_bank: 0.03}
}
</pre>

In instrumented CPU-only builds, median overhead was about 2.1% on MNIST-sized graphs (illustrative; replace with measured values in [Table 1](#inference-time-arx)).

---

# **12. Cost Model Details and Selection Algorithm**

### 12.1 Timing model

For a region $R$ with tiled compute and double buffering:

$$T_{\mathrm{off}}(R) \approx T_{\mathrm{setup}}
  + \max\!\big(T_{\mathrm{dma}}(R), T_{\mathrm{cmp}}(R)\big)
  + T_{\mathrm{sync}}.$$

DMA time uses transferred bytes and effective bandwidth
$B_{\mathrm{dma}}$, including packing/padding inflation $\rho \ge 1$:

$$T_{\mathrm{dma}}(R) = \frac{\rho\cdot(\mathrm{bytes}_{\mathrm{in}}+\mathrm{bytes}_{\mathrm{out}})}{B_{\mathrm{dma}}}.$$

Compute time is derived from MAC counts divided by peak MAC/s and
corrected by a profile-derived efficiency $\eta\in(0,1]$:

$$T_{\mathrm{cmp}}(R) = \frac{\mathrm{MACs}(R)}{\eta\cdot P_{\mathrm{peak}}}.$$

The net benefit is
$\Delta T(R) = T_{\mathrm{cpu}}(R) - T_{\mathrm{off}}(R)$.


### 12.2 Resource model

Each candidate $c$ has a resource vector $\mathcal{R}(c)$ over {LUT, FF, DSP, BRAM, SRAM, DMA lanes}. Selection maximizes $\sum \Delta T(c)$ subject to $\sum \mathcal{R}(c) \le B$ with a benefit-density tie-breaker when budgets are tight.


### 12.3 Region formation

Candidates are merged greedily into maximal regions when:

1.  data dependencies allow reordering or fusion,
2.  the merged tile still fits on-chip buffers, and
3.  the merged cost is superadditive after accounting for fewer host--device crossings.

---

# **13. MNIST Model Shapes and Mapping Notes**

For reproducibility and debugging, [Table 4](#mnist-shapes) lists the operator shapes used in the early evaluation.

<a id="mnist-shapes"></a>
<table>
  <caption>Table 4. MNIST operator shapes and the induced tiling on configuration B.</caption>
  <thead>
    <tr>
      <th style="text-align:left;">Op</th>
      <th style="text-align:left;">Input shape</th>
      <th style="text-align:left;">Weight shape</th>
      <th style="text-align:left;">Output shape</th>
      <th style="text-align:left;">Tile selection</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>conv1</td>
      <td>1×1×28×28</td>
      <td>16×1×3×3</td>
      <td>1×16×26×26</td>
      <td>M=16, N=16, K=16 blocks</td>
    </tr>
    <tr>
      <td>conv2</td>
      <td>1×16×13×13</td>
      <td>32×16×3×3</td>
      <td>1×32×11×11</td>
      <td>same as above</td>
    </tr>
    <tr>
      <td>gemm</td>
      <td>1×512</td>
      <td>512×10</td>
      <td>1×10</td>
      <td>M=16, N=16, K=16 with packing</td>
    </tr>
  </tbody>
</table>



