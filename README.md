### Official pytorch implementation of TiDE-Ab (ECCB 2026)

---

![TiDE-Ab Overview](assets/architecture.png)

> #### Title: &nbsp;&nbsp;&nbsp;&nbsp;&emsp; De Novo Epitope-Specific Antibody Design via Time-Dependent Guidance [[Paper](https://academic.oup.com/bioinformatics/article/42/Supplement_2/btag455/8767312)]<br>
>
> #### Authors: &nbsp; Yunji Kim, Minkyung Baek

#### Abstract

We propose TiDE-Ab, a conditional SE(3) flow matching framework for de novo epitope-specific antibody design. By conditioning on unpaired antigen and antibody structures without any pre-aligned frame, TiDE-Ab inherits the structural stability of flow matching while enabling global binding pose search from scratch. To further improve epitope targeting, we introduce Time-Dependent Classifier-Free Guidance (TD-CFG), which replaces static conditioning with an adaptive schedule: strong guidance early to establish the global binding pose, followed by gradual relaxation for precise local CDR refinement. On 55 non-redundant benchmark complexes, TiDE-Ab outperforms RFantibody with higher epitope recall (0.935 vs. 0.878) and over 95% fewer steric clashes. In therapeutic case studies on TGF-β and IL-17A, TiDE-Ab reproduced the binding profiles of clinical antibodies across isoform-selective and cross-reactive epitopes, whereas RFantibody consistently failed to produce viable candidates.

> **💡 Modality Support:**  
> In addition to standard antibodies (paired heavy and light chain Fv fragments), TiDE-Ab supports **VHH (single-domain antibodies / Nanobodies)**. The training set includes over 770 single-domain antibody complexes, allowing direct de novo design of VHH binders against specified epitopes. *(Note: Linear / scaffold-free peptides are not supported as the model requires an antibody framework).*

## 💊 Therapeutic Case Studies

TiDE-Ab demonstrates high practical utility by navigating complex therapeutic design scenarios that require programmable binding specificity. By successfully targeting a wide range of user-defined epitopes on the same antigen —from highly selective patches to broad conserved surfaces— TiDE-Ab ensures generated backbones strictly adhere to precise geometric constraints, transforming de novo design into a highly controllable and programmable engineering process.

<table align="center" width="95%">
  <tr valign="top">
    <td align="center" width="45%">
      <b> Case Study 1: TGF-β</b><br><br>
      <img src="assets/target1.png" width="100%"><br><br>
      <sub><b>TGF-β3 Selective vs. Pan-TGF-β</b></sub>
    </td>
    <td align="center" width="45%">
      <b> Case Study 2: IL-17A</b><br><br>
      <img src="assets/target2.png" width="100%"><br><br>
      <sub><b>IL-17A/F Dual vs. IL-17A Selective</b></sub>
    </td>
  </tr>
</table>

## ⚙️ Installation

[Pixi](https://pixi.sh/) is recommended for reproducible and fast environment management. These guides assume you are using **CUDA 12.4**.

### Option 1: Quick Install (Pixi) - _Recommended_

_Sets up the complete environment and dependencies automatically._

1. **Install Pixi** (if not already installed):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

2. **Install dependencies**:

```bash
pixi install
```

3. **Activate the environment** (optional, or run commands directly with `pixi run <task/command>`):

```bash
pixi shell
```

## 📂 Data Preparation

The model is trained on antibody-antigen complexes from the **SAbDab** database (cutoff: April 30, 2020). The dataset is managed through metadata files in `data/splits/`.

```bash
├── data
│   └── splits
│       ├── metadata_train.csv
│       ├── metadata_val.csv
│       └── metadata_test.csv
```

The metadata files in data/splits/ follow this schema:

| Column         | Description                                             |
| :------------- | :------------------------------------------------------ |
| **pdb_name**   | Unique identifier for the complex (e.g., `1yy9_D_C_A`). |
| **num_chains** | Total number of chains in the structure.                |
| **seq_len**    | Total sequence length of the complex.                   |
| **cluster**    | Interaction cluster ID used for balanced sampling.      |

## 🚀 Running the Code

You can run scripts either within `pixi shell` or directly using `pixi run`.

### Training

Run with default settings or override parameters via command line:

```bash
# Using pixi task:
pixi run train experiment.optimizer.lr=0.0005

# Or inside pixi shell / python directly:
python train.py experiment.optimizer.lr=0.0005  # (Optional) parameter override
```

### Inference

Pre-trained weights are located at `checkpoints/tideab_weights.pt` (used automatically by default).

#### Option A: Target Antigen De Novo Design (Recommended)
You can directly design standard antibodies (Fv) or single-domain antibodies (**VHH / Nanobodies**) against a target antigen PDB file with optional epitope/hotspot residues and sample count:

```bash
# Standard Antibody (Fv heavy + light chain scaffold)
pixi run inference --input demo/inputs/3eo1.pdb --output_dir results/

# VHH / Nanobody Design (single-domain heavy chain scaffold)
pixi run inference --input demo/inputs/3eo1.pdb --vhh --output_dir results/vhh_design/

# With custom hotspot residues (1-based on antigen chain) and sample count
pixi run inference --input demo/inputs/3eo1.pdb --vhh --hotspots 142 143 144 204 205 206 --num_samples 3 --output_dir results/3eo1_vhh_design/
```

Options available for `inference.py`:
- `--input, -i`: Path to the antigen PDB structure (e.g., `demo/inputs/3eo1.pdb`).
- `--vhh`: Use single-domain antibody (VHH / Nanobody) mode (automatically sets scaffold to `demo/inputs/vhh_scaffold.pdb`).
- `--hotspots, -hs`: (Optional) 1-based residue numbers specifying epitope hotspots.
- `--num_samples, -n`: Number of antibody candidates to generate (default: `1`).
- `--output_dir, -o`: Directory to save generated complexes and trajectories (default: `./results`).
- `--antibody, -ab`: (Optional) Custom antibody/VHH scaffold template PDB (default: `demo/inputs/hu-4D5-8_Fv.pdb`).
- `--weights_path, -w`: Path to model checkpoint (default: `checkpoints/tideab_weights.pt`).

#### Option B: Benchmark Test Set Evaluation
To evaluate on the full processed test dataset (SAbDab split):
```bash
pixi run inference --output_dir results/test_eval/
```

### Demo

We provide an easy‑to‑run demo for epitope‑specific antibody design in `run_demo.ipynb`. In this demo, you provide a PDB file containing your target antigen. The model then resamples the CDR loops while preserving the antibody framework structure, enabling de novo redesign of antibodies that bind the antigen in diverse binding modes. Both **standard Fv antibodies** (`demo/inputs/hu-4D5-8_Fv.pdb`) and **single-domain VHH nanobodies** (`demo/inputs/vhh_scaffold.pdb`) are supported.

To run the Jupyter notebook with Pixi:

```bash
pixi run jupyter
# Or launch run_demo.ipynb directly inside your preferred editor (VS Code, JupyterLab) with the `.pixi/envs/default` kernel.
```

## 🙏 Acknowledgements

This codebase is developed based on the [**FrameFlow**](https://github.com/microsoft/protein-frame-flow/tree/main) repository. We thank the original authors for their pioneering work on $SE(3)$ flow matching for protein structures.

This project is licensed under the MIT License - see the LICENSE file for details.
