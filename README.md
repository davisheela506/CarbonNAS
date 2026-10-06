# CarbonNAS: Carbon-Aware Multi-Objective Neural Architecture Search for Edge AI

A research project exploring how Neural Architecture Search (NAS) can be used to design AI models that balance **accuracy, latency, energy consumption, and carbon emissions**.

The goal is to build a small but reproducible research prototype that can automatically discover different neural network architectures and select an appropriate model depending on computational and carbon constraints.

This project is being developed as a practical introduction to **Carbon-Aware Neural Architecture Search**, with a focus on resource-efficient AI for edge computing.

---

## 1. Motivation

Modern AI systems are usually optimized mainly for predictive performance. In many real deployments, however, accuracy is only one part of the problem.

A model may be accurate but:

* take too long to run,
* consume too much energy,
* require more powerful hardware,
* or produce unnecessary carbon emissions.

These trade-offs become particularly important for AI systems deployed on edge devices and resource-constrained infrastructure.

This project investigates whether a set of neural networks can be automatically generated and optimized so that different models are available for different operating conditions.

Instead of searching for one model that is simply "the best", the project aims to find a **family of models with different accuracy, latency, energy and carbon characteristics**.

A runtime component can then select an appropriate model according to the current constraints.

---

# 2. Main Research Question

> Can Neural Architecture Search produce a family of edge-AI models that provides useful trade-offs between predictive accuracy, inference latency, energy consumption and carbon emissions?

The project will investigate this through a multi-objective NAS pipeline.

---

# 3. Project Objectives

The project will be developed in several stages.

### Objective 1 — Build a reproducible baseline

Start with a small image-classification problem and train several manually designed CNN models.

Measure:

* validation/test accuracy
* inference latency
* energy consumption
* estimated carbon emissions
* model size

This provides a baseline against which the NAS-generated architectures can be compared.

---

### Objective 2 — Define a NAS search space

Create a configurable CNN search space.

Possible architectural choices include:

* number of layers
* number of channels
* kernel size
* convolution type
* depthwise separable convolution
* expansion ratio
* pooling
* activation function
* skip connections

The search space should contain both small and relatively large models so that meaningful trade-offs can be observed.

---

### Objective 3 — Implement Neural Architecture Search

Implement an initial search strategy that generates and evaluates candidate architectures.

The first version will use a practical optimization method such as:

* random search
* Bayesian optimization with Optuna

A later version can investigate evolutionary search.

For every candidate architecture, the system will record its measured performance.

---

### Objective 4 — Add hardware-aware evaluation

The same architecture can behave differently on different hardware.

The project will therefore investigate model performance on available hardware, starting with CPU and GPU where possible.

The following measurements will be collected:

```text
Accuracy
Inference latency
Energy per inference
Model size
Throughput
```

If suitable hardware is available, the experiments can later be extended to an edge device such as an NVIDIA Jetson.

---

### Objective 5 — Estimate carbon emissions

Energy consumption alone does not completely describe the environmental impact of computation.

The project will therefore introduce a carbon-intensity factor.

The basic estimation will be:

```text
Carbon = Energy × Carbon Intensity
```

Different carbon-intensity scenarios will be considered to represent changing electricity-grid conditions.

The purpose is not to claim an exact real-world carbon footprint, but to study how changing carbon intensity affects model selection.

---

### Objective 6 — Implement multi-objective optimization

The NAS system will optimize several objectives simultaneously:

```text
Maximize:
    Accuracy

Minimize:
    Latency
    Energy
    Carbon
```

Rather than selecting one architecture using a single weighted score, the project will first identify **Pareto-optimal architectures**.

An architecture is Pareto-optimal when improving one objective would require sacrificing at least one of the other objectives.

---

### Objective 7 — Build a model family

The final NAS output should not be a single model.

Instead, the system should produce a collection of models with different characteristics.

For example:

Architecture A —
2 layers
32 channels
3×3 kernels

Architecture B —
3 layers
64 channels
3×3 kernels

Architecture C —
4 layers
32 channels
5×5 kernels

Architecture D —
3 layers
128 channels
depthwise convolutions
...

This creates a small model library that can be used by a runtime selection mechanism.

---

# 4. Carbon-Aware Runtime Model Selection

The final stage of the project will simulate a changing carbon-aware environment.

The runtime system will receive information such as:

```text
Current carbon intensity
Maximum acceptable latency
Minimum required accuracy
Energy budget
```

It will then select an appropriate model from the discovered model family.

For example:

```text
Low carbon intensity
        ↓
More computationally expensive model
        ↓
Higher accuracy


High carbon intensity
        ↓
More efficient model
        ↓
Lower energy and carbon
```

The goal is to demonstrate that the model does not have to remain fixed when the operating conditions change.

---

# 5. Proposed System

The complete system will follow this pipeline:

```text
                    Dataset
                       │
                       ▼
              ┌─────────────────┐
              │ NAS Search Space│
              └────────┬────────┘
                       │
                       ▼
              Candidate Architectures
                       │
                       ▼
              ┌──────────────────┐
              │ Model Training   │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Evaluation       │
              │                  │
              │ Accuracy         │
              │ Latency          │
              │ Energy           │
              │ Carbon           │
              └────────┬─────────┘
                       │
                       ▼
             Multi-Objective NAS
                       │
                       ▼
                Pareto Front
                       │
                       ▼
               Model Library
                       │
                       ▼
             Carbon-Aware Selector
                       │
                       ▼
              Runtime Model Choice
```

---

# 6. Dataset

The first experiments will use **CIFAR-10** because it is small enough to allow repeated experiments while still providing a meaningful image-classification task.

The project may later investigate another dataset if computational resources allow.

The dataset will be kept separate from the source code and downloaded automatically where possible.

---

# 7. Baseline Models

Before running NAS, several manually designed models will be trained.

The baseline should include models with different computational costs.

For example:

```text
Tiny CNN
Small CNN
Medium CNN
Larger CNN
```

For each model, record:

| Metric     | Measurement               |
| ---------- | ------------------------- |
| Accuracy   | %                         |
| Parameters | count                     |
| Model size | MB                        |
| Latency    | ms/inference              |
| Energy     | J/inference               |
| Carbon     | estimated gCO₂e/inference |

These baseline results will help determine whether the NAS process is actually discovering useful architectures.

---

# 8. NAS Search Space

A candidate architecture will be represented by a configuration similar to:

```text
number_of_layers
channels
kernel_size
operation
activation
expansion_ratio
skip_connection
```

A simplified example:

```python
architecture = {
    "layers": 4,
    "channels": [32, 64, 64, 128],
    "kernel_sizes": [3, 3, 5, 3],
    "operations": [
        "conv",
        "depthwise_conv",
        "conv",
        "depthwise_conv"
    ],
    "skip_connections": [True, False, True]
}
```

The exact search space will be refined during experimentation.

---

# 9. Search Strategy

### Phase 1 — Random Search

Start with random architecture generation.

This provides a simple baseline and helps understand the search space.

---

### Phase 2 — Bayesian Optimization

Use Optuna to investigate whether guided search can find good architectures with fewer evaluations.

The optimization will consider multiple objectives.

---

### Phase 3 — Evolutionary Search

If time and computational resources allow, implement an evolutionary NAS strategy.

Possible operations include:

```text
Mutation
Crossover
Selection
Population replacement
```

This will provide a stronger comparison between search strategies.

---

# 10. Energy Measurement

Energy measurement will depend on the available hardware.

Where direct power measurements are available, the project will measure power during inference.

A basic estimation is:

```text
Energy (J) = Average Power (W) × Execution Time (s)
```

Measurements should be repeated several times to reduce noise.

The project will report:

```text
mean
standard deviation
number of measurements
```

rather than relying on one measurement.

If direct hardware power measurement is not available, the project will clearly document the estimation method used.

---

# 11. Carbon Estimation

Carbon emissions will be estimated from energy consumption and a carbon-intensity value.

```text
Carbon = Energy × Carbon Intensity
```

Several scenarios will be evaluated:

```text
Low-carbon electricity
Medium-carbon electricity
High-carbon electricity
```

The purpose is to study how the preferred model changes when the carbon intensity of electricity changes.

The carbon-intensity assumptions will be documented clearly so that the experiments remain reproducible.

---

# 12. Pareto Analysis

The project will visualize the trade-offs between objectives.

Important plots will include:

### Accuracy vs Energy

```text
Accuracy
   ↑
   │       ●
   │     ●
   │   ●
   │ ●
   └────────────────→ Energy
```

### Accuracy vs Latency

```text
Accuracy
   ↑
   │       ●
   │     ●
   │   ●
   │ ●
   └────────────────→ Latency
```

### Accuracy vs Carbon

```text
Accuracy
   ↑
   │       ●
   │     ●
   │   ●
   │ ●
   └────────────────→ Carbon
```

The Pareto-optimal models will be highlighted.

---

# 13. Runtime Carbon-Aware Selection

The model selector will take runtime constraints such as:

```python
carbon_intensity
latency_budget
minimum_accuracy
energy_budget
```

and choose the most appropriate model from the model library.

Example:

```text
Carbon intensity = low
Latency budget = relaxed
Accuracy requirement = high

→ Select high-accuracy model
```

Another scenario:

```text
Carbon intensity = high
Latency budget = strict
Accuracy requirement = moderate

→ Select efficient model
```

The project will compare this adaptive strategy against using one fixed model for every condition.

---

# 14. Experiments

The final project should contain several experiments.

### Experiment 1 — Baseline

Compare manually designed CNNs.

### Experiment 2 — Random NAS

Search the architecture space randomly.

### Experiment 3 — Bayesian NAS

Use Optuna to guide the search.

### Experiment 4 — Multi-objective NAS

Generate the Pareto front using accuracy, latency, energy and carbon.

### Experiment 5 — Hardware comparison

Compare selected architectures across available hardware.

### Experiment 6 — Carbon-intensity scenarios

Change carbon intensity and observe how the preferred architectures change.

### Experiment 7 — Runtime model selection

Compare:

```text
Fixed model
        vs
Carbon-aware adaptive model
```

The final experiment should demonstrate whether adaptive selection can reduce energy/carbon while maintaining an acceptable level of accuracy.

---

# 15. Evaluation Metrics

The main metrics will be:

### Model quality

* Accuracy
* Validation loss

### Computational efficiency

* Number of parameters
* Model size
* FLOPs
* Inference latency
* Throughput

### Resource efficiency

* Energy per inference
* Energy per batch
* Estimated carbon per inference
* Estimated carbon per fixed number of inferences

### Search efficiency

* Number of architectures evaluated
* Search time
* Best accuracy found
* Pareto-front size

---

# 16. Expected Results

The project is not designed around a predetermined result.

Instead, the experiments should answer questions such as:

1. Does NAS find architectures that are more efficient than manually designed baselines?
2. How much accuracy is lost when reducing energy consumption?
3. How does hardware affect the ranking of architectures?
4. Does carbon intensity change which architecture is preferable?
5. Can a model family provide useful runtime trade-offs?
6. Does multi-objective search produce better trade-offs than optimizing accuracy alone?

The results will be reported even if some hypotheses are not supported.

---

# 17. Reproducibility

The project should be reproducible by another researcher.

The repository will therefore include:

* environment requirements
* configuration files
* fixed random seeds
* training scripts
* NAS search scripts
* evaluation scripts
* saved experiment results
* plotting scripts
* clear instructions for reproducing experiments

Where hardware-specific measurements are used, the exact hardware and software environment will be documented.

---

# 18. Project Structure

```text
carbon-aware-nas/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── configs/
│   ├── search_space.yaml
│   └── experiment.yaml
│
├── src/
│   ├── models/
│   │   ├── operations.py
│   │   ├── search_space.py
│   │   └── candidate_network.py
│   │
│   ├── nas/
│   │   ├── random_search.py
│   │   ├── bayesian_search.py
│   │   ├── evolutionary_search.py
│   │   └── pareto.py
│   │
│   ├── evaluation/
│   │   ├── accuracy.py
│   │   ├── latency.py
│   │   ├── energy.py
│   │   └── carbon.py
│   │
│   └── runtime/
│       └── model_selector.py
│
├── experiments/
│   ├── baseline.py
│   ├── random_nas.py
│   ├── bayesian_nas.py
│   ├── pareto_analysis.py
│   └── runtime_simulation.py
│
├── notebooks/
│   ├── 01_baseline_analysis.ipynb
│   ├── 02_nas_analysis.ipynb
│   └── 03_carbon_analysis.ipynb
│
├── results/
│   ├── architectures.csv
│   ├── pareto_front.csv
│   └── figures/
│
└── docs/
    └── methodology.md
```

---

# 19. Development Roadmap

## Week 1 — Foundations

* Set up the PyTorch environment
* Download CIFAR-10
* Implement the training pipeline
* Implement Tiny/Small/Medium CNN baselines
* Measure accuracy and latency
* Save all results automatically

**Deliverable:** working baseline experiments.

---

## Week 2 — NAS Search Space

* Design the architecture representation
* Implement configurable CNN generation
* Validate different architecture configurations
* Implement random architecture generation
* Train and evaluate candidate models

**Deliverable:** working NAS search space.

---

## Week 3 — Multi-Objective Evaluation

* Add parameter/FLOP calculation
* Add latency measurement
* Add energy measurement
* Add carbon estimation
* Store all measurements in structured CSV files

**Deliverable:** architecture evaluation pipeline.

---

## Week 4 — Multi-Objective NAS

* Implement Pareto-front calculation
* Implement random NAS baseline
* Integrate Optuna
* Run Bayesian optimization
* Compare search strategies

**Deliverable:** first Pareto-optimal architecture set.

---

## Week 5 — Carbon-Aware Runtime Selection

* Create the model library
* Implement the model-selection algorithm
* Simulate changing carbon intensity
* Add latency and accuracy constraints
* Compare adaptive selection with a fixed model

**Deliverable:** working carbon-aware model selector.

---

## Week 6 — Hardware Experiments

If hardware is available:

* benchmark CPU
* benchmark GPU
* benchmark edge hardware
* compare latency and energy
* analyze whether architecture rankings change across hardware

**Deliverable:** hardware-aware evaluation.

---

## Week 7 — Analysis

Generate:

* accuracy/energy plots
* accuracy/latency plots
* accuracy/carbon plots
* Pareto-front plots
* search-efficiency plots
* runtime model-selection plots

Perform statistical analysis and repeat important experiments.

**Deliverable:** complete experimental results.

---

## Week 8 — Research Prototype and Documentation

* Clean the repository
* Remove unnecessary code
* Add configuration files
* Add reproducibility instructions
* Document limitations
* Write the final README
* Add architecture diagrams
* Add result figures
* Prepare a short technical report

**Deliverable:** public research-quality GitHub repository.

---

# 20. What I Want This Project to Demonstrate

By the end of the project, the repository should demonstrate that I can:

* work with PyTorch
* design neural architectures
* understand Neural Architecture Search
* formulate multi-objective optimization problems
* evaluate models experimentally
* measure computational efficiency
* reason about energy consumption
* estimate carbon emissions
* analyze Pareto trade-offs
* work with edge-AI constraints
* build adaptive model-selection systems
* conduct reproducible research

The purpose is not to claim expertise in carbon-aware NAS from the beginning.

The purpose is to **learn the area by implementing a complete research prototype and evaluating it carefully**.

---

# 21. Limitations

This project will have several limitations.

The initial experiments will use a relatively small image-classification task rather than a production AI workload.

Carbon emissions will initially be estimated rather than measured directly from the electricity grid.

Hardware measurements will depend on the devices available.

The NAS search space will also be intentionally smaller than the search spaces used in large-scale industrial NAS systems.

These limitations will be explicitly discussed rather than hidden.

---

# 22. Possible Future Work

Several directions could extend the project:

* larger datasets
* larger NAS search spaces
* differentiable NAS
* evolutionary NAS
* hardware-specific search spaces
* real-time carbon-intensity APIs
* more accurate power measurement
* accelerator-aware optimization
* dynamic model switching
* reinforcement-learning-based NAS
* joint training and inference carbon optimization
* deployment on edge datacentres
* distributed NAS
* real-world workload evaluation

---

# 23. Final Goal

The final goal is to have a reproducible research prototype showing the following idea:

> **Instead of deploying one fixed AI model, design a family of models with different resource and accuracy characteristics and allow the system to select an appropriate model as computational and carbon conditions change.**

This project is intended as a practical exploration of **Carbon-Aware Neural Architecture Search and resource-efficient Edge AI**.

---
