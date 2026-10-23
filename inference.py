import os
import torch
import argparse
import logging
from hydra import compose, initialize
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from models.flow_module import FlowModule
from data.ab_datasets import AbDataModule
from utils import experiments_utils as eu
from utils import demo_utils

log = eu.get_pylogger(__name__)

def load_state_dict(weights_path, device='cpu'):
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    return state_dict

class DatasetPredictor:
    def __init__(self, cfg: DictConfig, weights_path: str, inference_dir: str):
        self._cfg = cfg
        self._module = FlowModule(self._cfg)
        self._module._inference_dir = inference_dir
        
        log.info(f"Loading weights from {weights_path}")
        state_dict = load_state_dict(weights_path)

        first_key = list(state_dict.keys())[0]
        if first_key.startswith('module.') and not hasattr(self._module.model, 'module'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        clean_state_dict = {}
        for k, v in state_dict.items():
            name = k
            while name.startswith('model.') or name.startswith('module.'):
                if name.startswith('model.'):
                    name = name.replace('model.', '', 1)
                if name.startswith('module.'):
                    name = name.replace('module.', '', 1)
            clean_state_dict[name] = v
        
        missing, unexpected = self._module.model.load_state_dict(clean_state_dict, strict=False)
        log.info(f"Missing keys : {missing}")
        log.info(f"Unexpected keys : {unexpected}")
        self._module.model.float()

        self._datamodule = AbDataModule(data_cfg=cfg.data)
        self._datamodule.setup(stage='test')

        self._trainer = Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            logger=False,
        )

    def predict(self):
        log.info("Starting Dataset Inference...")
        results = self._trainer.predict(
            model=self._module, 
            dataloaders=self._datamodule.test_dataloader()
        )
        return results

def run_pdb_inference(
    cfg: DictConfig,
    weights_path: str,
    antigen_pdb: str,
    output_dir: str,
    antibody_pdb: str = "./demo/inputs/hu-4D5-8_Fv.pdb",
    hotspots = None,
    num_samples: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Process antigen if needed (extract antigen chains if complex provided)
    if not antigen_pdb.endswith("_antigen.pdb"):
        log.info(f"Analyzing and preparing antigen chains from {antigen_pdb}...")
        demo_utils.create_antigen_pdb(antigen_pdb)
        processed_ag_pdb = antigen_pdb.replace('.pdb', '_antigen.pdb')
        if not os.path.exists(processed_ag_pdb):
            processed_ag_pdb = antigen_pdb
    else:
        processed_ag_pdb = antigen_pdb

    hotspot_residues = hotspots if hotspots is not None else []
    cdr_loop_lengths = {'H1': 7, 'H2': 6, 'H3': (5, 21), 'L1': (8, 13), 'L2': 7, 'L3': (9, 11)}

    log.info(f"Initializing Predictor with weights: {weights_path} (device: {device})")
    predictor = demo_utils.Predictor(cfg, weights_path=weights_path, device=device)

    log.info(f"Generating {num_samples} sample(s) for antigen: {processed_ag_pdb}")
    for bidx in range(num_samples):
        batch_, sample_ranges = demo_utils.process_model_input(
            antigen_pdb=processed_ag_pdb,
            hotspot_residues=hotspot_residues,
            antibody_pdb=antibody_pdb,
            cdr_loop_lengths=cdr_loop_lengths,
        )

        log.info(f"Sampling [{bidx + 1}/{num_samples}] ...")
        predictor.predict(batch_, bidx, output_dir, sample_ranges)

        sample_path = os.path.join(output_dir, f'sample_{bidx}.pdb')
        if os.path.exists(sample_path):
            if hotspot_residues:
                comment_parts = [f"C{res}" for res in hotspot_residues]
                comment_line = f"REMARK EPITOPE-LABEL: {', '.join(comment_parts)}\n"
                with open(sample_path, 'a') as f:
                    f.writelines(comment_line)
            try:
                demo_utils.quantitative_evaluation(batch_, sample_path)
            except Exception as e:
                log.warning(f"Evaluation warning for {sample_path}: {e}")

    log.info(f"Inference completed. Results saved to: {output_dir}")

def parse_args():
    default_weights = "checkpoints/tideab_weights.pt"
    if not os.path.exists(default_weights) and os.path.exists("weights.pt"):
        default_weights = "weights.pt"

    parser = argparse.ArgumentParser(description="TiDE-Ab Inference Script")
    
    # Input specification
    parser.add_argument(
        "--input", "-i",
        dest="input_pdb",
        type=str,
        default=None,
        help="Path to input target antigen PDB file (e.g., demo/inputs/3eo1.pdb). If not provided, runs on test dataset."
    )
    parser.add_argument(
        "--antibody", "-ab",
        dest="antibody_pdb",
        type=str,
        default="./demo/inputs/hu-4D5-8_Fv.pdb",
        help="Path to antibody template PDB (default: './demo/inputs/hu-4D5-8_Fv.pdb', or use './demo/inputs/vhh_scaffold.pdb' for VHH)"
    )
    parser.add_argument(
        "--vhh",
        action="store_true",
        default=False,
        help="Convenience flag: use the single-chain VHH nanobody scaffold ('demo/inputs/vhh_scaffold.pdb')"
    )
    parser.add_argument(
        "--hotspots", "-hs",
        nargs="*",
        type=int,
        default=None,
        help="List of epitope / hotspot residue numbers (1-based on antigen chain), e.g., --hotspots 142 143 144"
    )
    parser.add_argument(
        "--num_samples", "-n",
        type=int,
        default=1,
        help="Number of design samples to generate per input (default: 1)"
    )
    
    # Weights & output
    parser.add_argument(
        "--weights_path", "-w",
        type=str,
        default=default_weights,
        help=f"Path to model checkpoint (default: '{default_weights}')"
    )
    parser.add_argument(
        "--output_dir", "-o",
        dest="inference_dir",
        type=str,
        default="./results",
        help="Directory to save generated PDB samples (default: './results')"
    )

    # Positional backwards-compatibility: allows running `python inference.py <weights> <out_dir>`
    parser.add_argument(
        "pos_weights",
        nargs="?",
        type=str,
        default=None,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "pos_out_dir",
        nargs="?",
        type=str,
        default=None,
        help=argparse.SUPPRESS
    )

    return parser.parse_args()

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    args = parse_args()

    # Handle backwards-compatible positional arguments
    weights_path = args.pos_weights if args.pos_weights else args.weights_path
    inference_dir = args.pos_out_dir if args.pos_out_dir else args.inference_dir

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="base")

    if args.input_pdb is not None:
        if not os.path.exists(args.input_pdb):
            raise FileNotFoundError(f"Input antigen PDB not found: {args.input_pdb}")
        
        # If user did not pass hotspots, default to demo 3eo1 hotspots if demo input is used
        hotspots = args.hotspots
        if hotspots is None and "3eo1" in args.input_pdb:
            hotspots = [142, 143, 144, 204, 205, 206]
            log.info(f"Using default 3eo1 hotspot residues: {hotspots}")

        antibody_pdb = args.antibody_pdb
        if args.vhh:
            antibody_pdb = "./demo/inputs/vhh_scaffold.pdb"
            log.info("VHH mode selected: using single-domain scaffold ./demo/inputs/vhh_scaffold.pdb")

        run_pdb_inference(
            cfg=cfg,
            weights_path=weights_path,
            antigen_pdb=args.input_pdb,
            output_dir=inference_dir,
            antibody_pdb=antibody_pdb,
            hotspots=hotspots,
            num_samples=args.num_samples,
        )
    else:
        os.makedirs(inference_dir, exist_ok=True)
        predictor = DatasetPredictor(cfg, weights_path, inference_dir)
        _ = predictor.predict()
        log.info("Dataset inference finished.")

if __name__ == "__main__":
    main()