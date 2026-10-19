# About ChemSpace Copilot

**ChemSpace Copilot** is an AI-powered assistant for Generative Topographic Mapping (GTM) based chemical space analysis.

## System Architecture

ChemSpace Copilot uses a multi-agent system powered by DeepSeek LLM, coordinating specialized agents:

### Specialized Agents

| Agent | Purpose |
|-------|---------|
| **ChEMBL Downloader** | Downloads bioactivity data from ChEMBL database |
| **GTM Optimization** | Builds and optimizes Generative Topographic Maps |
| **GTM Density Analysis** | Analyzes compound distributions on GTM maps |
| **GTM Activity Analysis** | Creates activity-density landscapes for SAR |
| **GTM Loading** | Loads pre-existing GTM models from storage |
| **GTM Chemotype Analysis** | Analyzes scaffold distributions and chemotypes |
| **Autoencoder** | Molecular generation via the chembed autoencoder |
| **Autoencoder GTM Sampling** | Combines autoencoder with GTM for targeted generation |

## Capabilities

- **Data Retrieval**: Download chemical data from ChEMBL
- **Dimensionality Reduction**: Build GTM models for chemical space visualization
- **Property Analysis**: Map molecular properties onto chemical space landscapes
- **SAR Analysis**: Analyze activity distributions across chemical space
- **Molecular Generation**: Generate novel molecules using autoencoders
- **Chemotype Discovery**: Identify and analyze scaffold families

## Getting Started

Type your request in natural language:
- "Download bioactivity data for EGFR from ChEMBL"
- "Build a GTM model from the downloaded data"
- "Show activity landscape for IC50 values"
- "Generate molecules similar to aspirin"

## File Upload

Upload molecular data files (CSV, SDF) directly in the chat for analysis.

---

## About Chemography

**Chemography** is the science of mapping chemical space using dimensionality reduction techniques, particularly Generative Topographic Mapping (GTM).

### What is GTM?

Generative Topographic Mapping is a probabilistic dimensionality reduction method that projects high-dimensional molecular descriptor space onto a 2D manifold, enabling visualization and analysis of chemical space.

### Key Concepts

- **Chemical Space** — The abstract multidimensional space of all possible molecules
- **Molecular Descriptors** — Numerical representations of molecular properties
- **Activity Landscapes** — Maps showing how biological activity varies across chemical space
- **Chemotypes** — Families of molecules sharing common scaffolds

### ✨ Applications of GTM in Chemography

Chemography turns high-dimensional molecular data into navigable maps. With **Generative Topographic Mapping (GTM)**, these maps are probabilistic, interpretable, and directly useful for medicinal chemistry workflows.

#### 🧭 1) Chemical Space Navigation
- Organize millions of compounds on a smooth 2D manifold
- Detect global chemotype neighborhoods and local subseries clusters
- Compare libraries, projects, or campaigns in a single reference space

#### 🧪 2) Structure–Activity Landscape Analysis
- Project potency, selectivity, and ADMET endpoints onto GTM landscapes
- Reveal activity cliffs, smooth SAR trends, and underexplored regions
- Support hypothesis generation before expensive synthesis cycles

#### 🎯 3) Virtual Screening & Hit Prioritization
- Identify map zones enriched in known actives
- Rank candidates by neighborhood context rather than single-point similarity
- Focus triage on compounds balancing novelty and expected activity

#### 🧬 4) Chemotype Intelligence
- Quantify scaffold occupancy and chemotype diversity over the map
- Track which structural families dominate or are missing in a dataset
- Guide scaffold hopping toward productive but less crowded regions

#### 🚀 5) Lead Optimization Strategy
- Monitor how analog series move through chemical space over iterations
- Connect structural edits with shifts in potency/property landscapes
- Prioritize directions likely to improve multiparameter profiles

#### 🤖 6) Generative Design with Spatial Control
- Couple molecular generators with GTM coordinates to sample targeted zones
- Steer generation toward desirable activity/property neighborhoods
- Filter ideas by map-consistency for better design robustness

#### 📊 7) Portfolio-Level Decision Support
- Benchmark projects using comparable GTM representations
- Detect redundancy across internal collections
- Highlight white spaces for strategic exploration

> **In short:** GTM-powered chemography is not just visualization—it is a decision framework for discovery, optimization, and molecular innovation.

> **Reference:** Bishop, C. M., Svensén, M., & Williams, C. K. (1998). GTM: The generative topographic mapping. *Neural computation*, 10(1), 215-234.
