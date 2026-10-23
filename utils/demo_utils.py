
import dataclasses
import numpy as np
import copy
import py3Dmol
import os
import re
from collections import OrderedDict
from anarci import anarci
from io import StringIO
from sklearn.metrics import precision_recall_fscore_support

import torch
from data import utils as du
from analysis import utils as au
from utils import experiments_utils as eu

from data import parsers
import data.protein as dprotein
from openfold.utils.rigid_utils import Rotation, Rigid
from models.flow_module import FlowModule

from Bio.SeqUtils import seq1
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, NeighborSearch, Superimposer
from Bio.PDB import PDBParser, Superimposer

PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
_rg_rots = torch.eye(3, dtype=torch.float64)
_rg_rots[0, 0] = -1
_rg_rots[2, 2] = -1



def analyze_complex(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('target', pdb_path)
    results = {}

    for model in structure:
        for chain in model:
            chain_id = chain.get_id()

            residues = [res for res in chain if res.get_id()[0] == ' ']
            sequence = "".join([seq1(res.get_resname()) for res in residues])

            if not sequence:
                continue

            out = anarci([("query", sequence)], scheme="chothia")
            
            if out[0][0] is None or out[1][0] is None:
                results[chain_id] = {'type': 'antigen'}
                continue

            actual_numbering = out[0][0][0] 
            details = out[1][0][0]

            raw_type = details.get('type') or details.get('chain_type')
            if raw_type is None:
                results[chain_id] = {'type': 'antigen'}
                continue

            chain_type = 'H' if raw_type == 'H' else 'L'

            results[chain_id] = {
                'type': chain_type,
                'numbering': actual_numbering,
                'residues': residues[details['query_start'] : details['query_end'] + 1],
                'details': details
            }

    antigen_chains = [cid for cid, info in results.items() if info['type'] == 'antigen']

    return antigen_chains



def create_antigen_pdb(input_pdb):
    """
    Extract antigen chains and create a new PDB file with renumbered indices.
    All chains will have ID '2' and residues will be numbered sequentially from 1.

    Args:
        input_pdb: Path to input PDB file
    """

    antigen_chains = analyze_complex(input_pdb)
    output_pdb = input_pdb.replace('.pdb', '_antigen.pdb')

    if not antigen_chains:
        print("No antigen chains found.")
        return

    print(f"Found {len(antigen_chains)} antigen chain(s): {antigen_chains}")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('original', input_pdb)
    model = structure[0]

    new_structure = Structure.Structure('processed')
    new_model = Model.Model(0)
    new_structure.add(new_model)

    new_chain = Chain.Chain('C')
    residue_counter = 1
    for cid in antigen_chains:
        if cid not in model:
            print(f"Chain {cid} not found in structure.")
            continue
        original_chain = model[cid]
        for residue in original_chain:
            new_residue = residue.copy()
            old_id = new_residue.get_id()
            new_id = (old_id[0], residue_counter, ' ')
            new_residue.id = new_id
            new_chain.add(new_residue)
            residue_counter += 1

    new_model.add(new_chain)

    io = PDBIO()
    io.set_structure(new_structure)
    io.save(output_pdb)
    print(f"Antigen-only PDB saved to: {output_pdb}")



def visualize_antigen_with_selection(pdb_path):
    with open(pdb_path, 'r') as f:
        pdb_data = f.read()

    view = py3Dmol.view(width=800, height=600)
    view.addModel(pdb_data, 'pdb')
    view.setStyle({'cartoon': {'color': 'grey'}})
    view.zoomTo()

    callback = """
    function(atom, viewer) {
        if(!atom) return;
        console.log("Selected Residue: " + atom.chain + "_" + atom.resi);
        atom.clicked = !atom.clicked;
        if(atom.clicked) {
            viewer.addLabel(atom.chain + ":" + atom.resi, 
                {position: atom, backgroundColor: 'red', fontColor: 'white'});
        } else {
            viewer.removeLabels({resn: atom.resn, chain: atom.chain, resi: atom.resi});
        }
    }
    """
    view.setClickable({}, True, callback)
    return view



def _read_cdr_from_pdb(pdb_path):
    cdr_info = {'H1': [], 'H2': [], 'H3': [], 'L1': [], 'L2': [], 'L3': []}
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('REMARK PDBinfo-LABEL:'):
                parts = line.strip().split()
                if len(parts) >= 3:
                    res_num = int(parts[2])
                    loop_name = parts[3]
                    if loop_name in cdr_info:
                        cdr_info[loop_name].append(res_num)
    return cdr_info



def _concat_features(struct_feats):
    concat_dict = {}
    for key in struct_feats[0].keys():
        concat_dict[key] = np.concatenate([f[key] for f in struct_feats], axis=0)
    return concat_dict



def process_model_input(antigen_pdb, cdr_loop_lengths, antibody_pdb, hotspot_residues):

    # Parse structures
    cdr_info = _read_cdr_from_pdb(antibody_pdb)
    parser = PDBParser(QUIET=True)

    # Prepare antibody chains and antigen chains
    ab_struct = parser.get_structure('ab', antibody_pdb)
    ab_orig_chains = list(ab_struct[0].get_chains())
    is_vhh = len(ab_orig_chains) == 1

    ab_chains_renamed = []
    if is_vhh:
        # VHH: single heavy chain renamed to 'A'
        chain = ab_orig_chains[0]
        new_chain = Chain.Chain('A')
        for idx, residue in enumerate(chain.get_residues(), start=1):
            new_res = residue.copy()
            new_res.id = (' ', idx, ' ')
            new_chain.add(new_res)
        ab_chains_renamed.append(new_chain)
    else:
        # Standard antibody (Fv): H->A, L->B
        chain_map = {'H': 'A', 'L': 'B'}
        for orig_chain in ab_orig_chains:
            new_chain_id = chain_map.get(orig_chain.id, orig_chain.id)
            new_chain = Chain.Chain(new_chain_id)
            for idx, residue in enumerate(orig_chain.get_residues(), start=1):
                new_res = residue.copy()
                new_res.id = (' ', idx, ' ')
                new_chain.add(new_res)
            ab_chains_renamed.append(new_chain)

    ag_struct = parser.get_structure('ag', antigen_pdb)
    ag_chains = list(ag_struct[0].get_chains())

    # Extract and concatenate chain features
    # For VHH: antibody=0, antigen=1. For standard Fv: antibody=(0, 1), antigen=2
    target_chain_idx = 1 if is_vhh else 2
    all_chains = ab_chains_renamed + ag_chains
    struct_feats = []
    for i, chain in enumerate(all_chains):
        c_idx = i if i < target_chain_idx else target_chain_idx
        struct_feats.append(dataclasses.asdict(parsers.process_chain(chain, c_idx)))
    complex_feats = _concat_features(struct_feats)

    # Calculate lengths and create masks
    n_total_res = len(complex_feats['residue_index'])
    ab_total_len = sum(len(list(c.get_residues())) for c in ab_chains_renamed)

    all_cdr_residues = [res for res_list in cdr_info.values() for res in res_list]
    framework_mask = np.ones(n_total_res, dtype=bool)
    for res_num in all_cdr_residues:
        framework_mask[res_num - 1] = False

    hotspot = np.zeros(n_total_res)
    for res_num in hotspot_residues:
        hotspot[ab_total_len + res_num - 1] = 1

    diffuse_mask = np.zeros(n_total_res, dtype=bool)
    diffuse_mask[:ab_total_len] = 1

    # Center positions using hotspot center of mass
    all_atom_pos = torch.tensor(complex_feats['atom_positions']).double()
    bb_pos = all_atom_pos[:, 1, :]
    hotspot_tensor = torch.tensor(hotspot)
    center = (bb_pos * hotspot_tensor[:, None]).sum(0) / (hotspot_tensor.sum() + 1e-8)
    all_atom_pos -= center[None, None, :]

    # Calculate rigid transforms
    rigids_1 = Rigid.from_3_points(all_atom_pos[:, 2, :], all_atom_pos[:, 1, :], all_atom_pos[:, 0, :])
    rigids_1 = rigids_1.compose(Rigid(Rotation(rot_mats=_rg_rots), None))
    rotmats_1 = rigids_1.get_rots().get_rot_mats()
    trans_1 = rigids_1.get_trans()

    # Build base features
    aatype = torch.tensor(complex_feats['aatype'])
    feats = {
        'aatype': aatype.float(),
        'atom_pos': all_atom_pos[..., :14, :].float(),
        'atom_mask': torch.tensor(complex_feats['atom_mask'][..., :14]),
        'rotmats_1': rotmats_1,
        'trans_1': trans_1,
        'diffuse_mask': torch.tensor(diffuse_mask).float(),
        'framework_mask': torch.tensor(framework_mask).float(),
        'res_mask': torch.ones_like(aatype).float(),
        'chain_index': torch.tensor(complex_feats['chain_index']).float(),
        'residue_index': torch.tensor(complex_feats['residue_index']).int(),
        'hotspot_1d': hotspot_tensor.float(),
    }


    # Extract framework/CDR ranges and generate sample ranges
    framework_indices = torch.where(feats['framework_mask'].bool())[0].numpy()
    cdr_indices = torch.where(~feats['framework_mask'].bool())[0].numpy()
    framework_ranges = eu.ranges(framework_indices)
    cdr_ranges = eu.ranges(cdr_indices)

    cdr_names = ['H1', 'H2', 'H3'] if is_vhh else ['H1', 'H2', 'H3', 'L1', 'L2', 'L3']
    sample_ranges = {}
    for idx, cdr_name in enumerate(cdr_names):
        if idx < len(cdr_ranges) and cdr_name in cdr_loop_lengths:
            start_idx, end_idx = cdr_ranges[idx]
            length_spec = cdr_loop_lengths[cdr_name]
            target_len = np.random.randint(*length_spec) if isinstance(length_spec, tuple) else length_spec
            sample_ranges[cdr_name] = [(start_idx, end_idx), target_len]
    
    ordered_sample_ranges = OrderedDict()
    fr_idx = prev_idx = 0
    for cdr_key in cdr_names:
        if cdr_key in sample_ranges:
            cdr_start = sample_ranges[cdr_key][0][0]
            if fr_idx < len(framework_ranges):
                fr_start, fr_end = framework_ranges[fr_idx]
                if fr_start == prev_idx:
                    ordered_sample_ranges[f'F_{fr_idx}'] = [(fr_start, fr_end)]
                    prev_idx = fr_end + 1
                    fr_idx += 1
            ordered_sample_ranges[cdr_key] = sample_ranges[cdr_key]
            prev_idx = sample_ranges[cdr_key][0][1] + 1

    if fr_idx < len(framework_ranges):
        ordered_sample_ranges[f'F_{fr_idx}'] = [framework_ranges[fr_idx]]
    
    return get_input_w_rand_len(feats, 'cuda', ordered_sample_ranges)


def repeat_first_dim(x, times):
    return x.repeat((times,) + (1,) * (x.ndim - 1))


def get_input_w_rand_len(data_, device, sample_ranges):
    """Convert input batch with random CDR lengths."""
    data = copy.deepcopy(data_)
    batch_ = {k: [] for k in data}
    for smp_k in sample_ranges.keys():
        if smp_k[0] == 'F':
            (start_idx, end_idx) = sample_ranges[smp_k][0]
            for k in batch_:
                batch_[k].append(data[k][start_idx:end_idx+1])
        else:
            (start_idx, end_idx), fill_len = sample_ranges[smp_k]
            for k in batch_:
                batch_[k].append(repeat_first_dim(data[k][start_idx:start_idx+1], fill_len))
                
    batch_ = {k: torch.cat(batch_[k]).float().to(device) for k in batch_}
    batch_['aatype'][~batch_['framework_mask'].bool()] = 20
    batch_['residue_index'] = torch.cat([torch.arange(1, torch.sum(batch_['chain_index'] == cidx) + 1) 
                                         for cidx in torch.unique(batch_['chain_index'])])
    batch_['chain_idx_oh'] = torch.nn.functional.one_hot(batch_['chain_index'].long(), 3).float()
    batch_['aatype_oh'] = torch.nn.functional.one_hot(batch_['aatype'].long(), 21).float()
    batch_['hotspot_2d'] = batch_['hotspot_1d'][:, None] * batch_['hotspot_1d'][None, :]
    converted_cdr_ranges = eu.ranges(torch.where(~batch_['framework_mask'].cpu().bool())[0].numpy())

    converted_sample_ranges = {}
    cv_cdr_idx = 0
    for k in sample_ranges.keys():
        if k[0] != 'F':
            a, b = converted_cdr_ranges[cv_cdr_idx]
            converted_sample_ranges[k] = batch_['residue_index'][a:b+1].cpu().numpy().tolist()
            cv_cdr_idx += 1

    return batch_, converted_sample_ranges



def quantitative_evaluation(batch_, sample_path):

    seq_len = batch_['res_mask'].shape[0]
    chain_indices = batch_['chain_index'].cpu().numpy()

    # Determine antibody boundary: diffused residues belong to the designed antibody/VHH
    diffuse_mask = batch_['diffuse_mask'].cpu().numpy().astype(bool)
    num_ab_res = diffuse_mask.sum()
    num_ag_res = seq_len - num_ab_res

    gt_hotspot = batch_['hotspot_1d'][num_ab_res:].cpu().numpy()
    loop_mask = (1-batch_['framework_mask']).cpu().numpy()
    loop_indices = np.where(loop_mask)[0].tolist() 
    loop_segments = eu.ranges(loop_indices)

    pred_data = du.get_gt14_value_from_pdb(sample_path, 4)
    pred_bb_positions = torch.tensor(pred_data['bb_positions'])
    pred_chain_index = pred_data['chain_index']

    # In prediction PDB, antibody is chain A (and B if Fv); antigen chains follow
    pred_num_ab_res = num_ab_res

    pred_ab_pos = pred_bb_positions[:pred_num_ab_res]
    pred_ag_pos = pred_bb_positions[pred_num_ab_res:]

    dist_matrix = torch.cdist(pred_ab_pos, pred_ag_pos)  # [N_ab, N_ag]
    min_ab_ag_distance = torch.min(dist_matrix).item()
    print(f"Minimum antibody-antigen distance: {min_ab_ag_distance:.2f} Å")


    i, j = torch.where(dist_matrix < 10)
    pred_hotspot = torch.zeros(pred_ag_pos.shape[0])
    pred_hotspot[j] = 1
    metrics = precision_recall_fscore_support(gt_hotspot, pred_hotspot.numpy(), average='binary')[:3]
    print(f"Hotspot Precision: {metrics[0]:.4f}, Recall: {metrics[1]:.4f}")


    gt_atom_pos = batch_['atom_pos'].cpu().numpy()
    residue_index = batch_['residue_index'].cpu().numpy()

    gt_pdb = au.create_full_prot(
        gt_atom_pos, 
        np.ones(gt_atom_pos.shape[:2]), 
        chain_indices,
        np.zeros(seq_len).astype(np.int8), 
        residue_index
    )
    gt_pdb = dprotein.to_pdb(gt_pdb, model=1, add_end=False)

    gt_framework = batch_['framework_mask'].bool() * batch_['diffuse_mask'].bool()
    gt_framework_ranges = eu.ranges(torch.where(gt_framework)[0].cpu().numpy())
    gt_framework_patches = np.concatenate(
        [gt_atom_pos[s:e+1, :, :] for (s, e) in gt_framework_ranges]
    )[:, 1, :]  # CA atoms only

    pred_data = du.get_gt14_value_from_pdb(sample_path)
    pred_pdb = au.create_full_prot(
        pred_data['atom_positions'],
        np.ones(pred_data['atom_positions'].shape[:2]),
        pred_data['chain_index'],
        np.zeros(pred_data['chain_index'].shape[0]).astype(np.int8),
        pred_data['residue_index']
    )
    pred_pdb = dprotein.to_pdb(pred_pdb, model=1, add_end=False)
        
    # Alignment
    pred_pdb = align_coords_rf(
        gt_pdb, pred_pdb, ['A', 'B'], 
        gt_framework, gt_framework
    )

    # Aligned predicted positions
    pred_atom_pos = du.get_gt14_value_from_pdb(
        pred_pdb, 4, centering=False
    )['atom_positions']
    pred_framework_patches = np.concatenate(
        [pred_atom_pos[s:e+1, :, :] for (s, e) in gt_framework_ranges]
    )[:, 1, :]  # CA atoms only

    diff = pred_framework_patches - gt_framework_patches
    rmsd = np.sqrt((diff * diff).sum() / diff.shape[0])
    print(f"Framework RMSD: {rmsd}")


    clash_count, clash_res = count_all_atom_clashes(sample_path, 'bb')
    print(f"Number of Clashing Residues: {clash_count} at {clash_res}")


    with open(sample_path, 'a') as f:
        f.write(f"Hotspot Precision: {metrics[0]:.4f}, Recall: {metrics[1]:.4f}" + '\n')
        f.write(f"Minimum antibody-antigen distance: {min_ab_ag_distance:.2f} Å" + '\n')
        f.write(f"Framework RMSD: {rmsd}" + '\n')
        f.write(f"Number of Clashing Residues: {clash_count}" + '\n')



def align_coords_rf(gt_pdb, pred_pdb, chains, gt_frmask, pred_frmask):
    parser = PDBParser(QUIET=True)
    gt_str = parser.get_structure("reference", StringIO(gt_pdb))
    pred_str = parser.get_structure("reference", StringIO(pred_pdb))

    ## align antibodies
    gt_atoms = []
    pred_atoms = []
    gt_fridx = 0
    pred_fridx = 0
    for _idx, chain_idx in enumerate(chains):
        gt_atoms += [res['CA'] for i, res in enumerate(gt_str[0][chain_idx]) if 'CA' in res and gt_frmask[gt_fridx+i]]
        pred_atoms += [res['CA'] for i, res in enumerate(pred_str[0][chain_idx]) if 'CA' in res and pred_frmask[pred_fridx+i]] 
        gt_fridx += len(gt_str[0][chain_idx])
        pred_fridx += len(pred_str[0][chain_idx])

    super_imposer = Superimposer()
    super_imposer.set_atoms(gt_atoms, pred_atoms)
    super_imposer.apply(pred_str.get_atoms())
    io = PDBIO()
    io.set_structure(pred_str)
    output = StringIO()
    io.save(output)
    pred_pdb = output.getvalue()
    return pred_pdb



def count_all_atom_clashes(pdb_file, mode, clash_tolerance = 0.3):
    vdw_radii = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'H': 1.20, 'P': 1.80}
    default_radius = 1.50

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("prot", pdb_file)
    
    atoms = []
    for chain in struct[0]:
        for residue in chain:
            if residue.id[0] != ' ': continue
            for atom in residue:
                if mode == 'bb':
                    if atom.get_name() not in ['N', 'CA', 'C']:
                        continue
                if atom.get_altloc() not in [' ', 'A']:
                    continue
                if atom.element == 'H':
                    continue
                atoms.append(atom)

    ns = NeighborSearch(atoms)
    pairs = ns.search_all(5.0) 
    
    clashing_residues = set()
    
    for a1, a2 in pairs:
        res1 = a1.get_parent()
        res2 = a2.get_parent()
        c1 = res1.get_parent().id
        c2 = res2.get_parent().id
        
        if c1 == c2:
            res_diff = abs(res1.id[1] - res2.id[1])
            if res_diff <= 3:
                continue

        elem1 = a1.element.strip().upper()
        elem2 = a2.element.strip().upper()
        
        r1 = vdw_radii.get(elem1, default_radius)
        r2 = vdw_radii.get(elem2, default_radius)
        vdw_sum = r1 + r2
        
        dist = a1 - a2
        
        if dist < (vdw_sum - clash_tolerance):
            clashing_residues.add((c1, res1.id[1]))
            clashing_residues.add((c2, res2.id[1]))
            
    unique_count = len(clashing_residues)
    return unique_count, sorted(list(clashing_residues))



def parse_labels_from_pdb(pdb_path):
    cdr_residues = set()
    epitope_residues = set()

    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("REMARK") and "PDBinfo-LABEL:" in line:
                parts = line.split()
                res_num = int(parts[2])
                chain_id = parts[3]
                cdr_residues.add((chain_id, res_num))

            elif line.startswith("REMARK") and "EPITOPE-LABEL:" in line:
                res_str = line.split("EPITOPE-LABEL:")[-1].strip()
                for item in res_str.split(","):
                    item = item.strip()
                    m = re.match(r"([A-Za-z])(\d+)", item)
                    if m:
                        chain_id = m.group(1)
                        res_num = int(m.group(2))
                        epitope_residues.add((chain_id, res_num))

    return cdr_residues, epitope_residues



def visualize_output(sample_path: str, width=1000, height=600):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(
        os.path.basename(sample_path).split("_")[0], sample_path
    )

    cdr_residues, epitope_residues = parse_labels_from_pdb(sample_path)

    grid = (1, 1)
    view = py3Dmol.view(width=width, height=height, linked=True, viewergrid=grid)

    with open(sample_path, "r") as f:
        sample_pdb_string = f.read()
    view.addModel(sample_pdb_string, "pdb", viewer=(0, 0))

    view.setStyle({}, {"cartoon": {"color": "#CCCCCC"}}, viewer=(0, 0))

    for chain in structure[0]:
        chain_id = chain.id

        for residue in chain:
            res_num = residue.id[1]

            is_epitope = (chain_id, res_num) in epitope_residues
            is_cdr = (chain_id, res_num) in cdr_residues
            # For VHH, antibody is chain A; for Fv, chains A and B
            ab_chains = {"A", "B"}
            is_ab = chain_id in ab_chains

            if is_epitope:
                color = "#DC143C"  # Epitope (Red)
            elif is_ab and is_cdr:
                color = "#FFD700"  # CDR (Yellow)
            elif is_ab and not is_cdr:
                color = "#EEDBC1"  # Antibody Framework (Beige)
            else:
                color = "#4A4A4A"  # Antigen (Gray)

            view.setStyle(
                {"chain": chain_id, "resi": str(res_num)},
                {"cartoon": {"color": color}},
                viewer=(0, 0),
            )

    view.zoomTo()
    view.setBackgroundColor("white")
    return view.show()



class Predictor:
    
    def __init__(self, cfg, weights_path, device='cuda'):
        self.device = device
        self.module = FlowModule(cfg)
        
        state_dict = self._load_weights(weights_path)
        self.module.model.load_state_dict(state_dict, strict=False)
        self.module.model.eval()
        self.module.model.to(device)
    
    def _load_weights(self, weights_path):
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        clean_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '').replace('model.', '')
            clean_dict[name] = v
        return clean_dict
    
    @torch.no_grad()
    def predict(self, model_input, batch_idx, sample_dir, sample_ranges):
        if isinstance(model_input, dict):
            model_input = {k: v.unsqueeze(0).to(self.device) if torch.is_tensor(v) else v for k, v in model_input.items()}
        return self.module.predict_step(model_input, batch_idx, True, sample_dir, sample_ranges)



def extract_cdr_residues_from_file(filename):
    residues = []
    with open(filename, 'r') as f:
        for line in f:
            if 'PDBinfo-LABEL:' in line:
                parts = line.split()
                res_num = parts[2]
                chain_id = parts[3]
                residues.append(f"{chain_id}{res_num}")
    return ' '.join(residues)