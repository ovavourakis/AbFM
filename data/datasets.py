import tree
import math
import random
import logging
import itertools
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import dist

from data import utils as du

from openfold.data import data_transforms
from openfold.utils import rigid_utils


class PdbDataset(Dataset):
    """
    Dataset of antibody features derived from PDBs, stored as 
    pre-processed pickles (see process_ab_pdb_files.py).

    Dataset passed as fully pre-filtered pandas DataFrame.
    """
    def __init__(self, *, pdb_csv):
        self._log = logging.getLogger(__name__)
        self.pdb_csv = pdb_csv
        
    def _process_csv_row(self, processed_file_path):
        '''Load and process a single row of the PDB CSV file.'''
        processed_feats = du.read_pkl(processed_file_path)
        processed_feats = du.parse_chain_feats(processed_feats)

        # only take modeled residues (natural AAs, not others)
        modeled_idx = processed_feats['modeled_idx']
        min_idx, max_idx = np.min(modeled_idx), np.max(modeled_idx)
        del processed_feats['modeled_idx']
        processed_feats = tree.map_structure(lambda x: x[min_idx:(max_idx+1)], 
                                             processed_feats)

        # run through OpenFold data transforms
        chain_feats = {
            'aatype': torch.tensor(processed_feats['aatype']).long(),
            'all_atom_positions': torch.tensor(processed_feats['atom_positions']).double(), # non-centred positions
            'all_atom_mask': torch.tensor(processed_feats['atom_mask']).double()
        }
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
        rotmats_1 = rigids_1.get_rots().get_rot_mats()
        trans_1 = rigids_1.get_trans()
        res_idx = processed_feats['residue_index']

        # re-index residues starting at 1 (heavy chain) and 1001 (light chain), preserving gaps
        light_chain_start = np.argmax(res_idx >= 1000)
        heavy_chain_res_idx = res_idx[:light_chain_start] - np.min(res_idx[:light_chain_start]) + 1
        light_chain_res_idx = res_idx[light_chain_start:] - np.min(res_idx[light_chain_start:]) + 1 + 1000
        res_idx = np.concatenate([heavy_chain_res_idx, light_chain_res_idx])

        return {
            'aatype': chain_feats['aatype'],
            'res_idx': res_idx,
            'rotmats_1': rotmats_1,
            'trans_1': trans_1,
            'res_mask': torch.tensor(processed_feats['bb_mask']).int(),
        }

    def __len__(self):
        return len(self.pdb_csv)

    def __getitem__(self, idx):
        # return one structure from pdb csv
        csv_row = self.pdb_csv.iloc[idx]
        chain_feats = self._process_csv_row(csv_row['processed_path'])
        chain_feats['csv_idx'] = torch.ones(1, dtype=torch.long) * idx
        return chain_feats
    
class LengthDataset(Dataset):
    """
    Dummy dataset of just antibody params from which to
    generate de novo samples. Each param has the form 
    (num_res_heavy, num_res_light, sample_id).

    Entries are created based on a samples configuration passed
    to the constructor. 

    Chain length combinations are sampled uniformly from the set 
    of all permissible combinations such that the total length
    does not exceed max_length or fall below min_length
    (see config).
    """
    def __init__(self, samples_cfg):
        self._samples_cfg = samples_cfg
        random.seed(samples_cfg.seed)

        if samples_cfg.length_list is not None:
            all_sample_lengths = [int(x) for x in samples_cfg.length_list]
        else:
            all_sample_lengths = range(self._samples_cfg.min_length,
                                       self._samples_cfg.max_length+1,
                                       self._samples_cfg.length_step
            )
        # all permissible light/heavy-chain length combinations per total length
        self.length_combos = {}
        for len_h in range(self._samples_cfg.min_length_heavy, self._samples_cfg.max_length_heavy + 1):
            for len_l in range(self._samples_cfg.min_length_light, self._samples_cfg.max_length_light + 1):
                total_length = len_h + len_l
                if self._samples_cfg.min_length <= total_length <= self._samples_cfg.max_length:
                    if total_length not in self.length_combos:
                        self.length_combos[total_length] = []
                    self.length_combos[total_length].append((len_h, len_l))
        
        # sample length combos for each length
        all_sample_ids = []
        for length in all_sample_lengths:
            len_combos = random.choices(self.length_combos[length], 
                                        k=self._samples_cfg.samples_per_length)
            for sample_id, len_combo in enumerate(len_combos):
                all_sample_ids.append((len_combo[0], len_combo[1], sample_id))
        self._all_sample_ids = all_sample_ids

    def __len__(self):
        return len(self._all_sample_ids)

    def __getitem__(self, idx):
        # idx must be a single integer
        len_h, len_l, sample_id = self._all_sample_ids[idx]
        batch = {
            'len_h' : len_h,
            'len_l' : len_l,
            'sample_id': sample_id,
        }
        return batch
    
class CombinedDataset(Dataset):
    """
    Combines a PDBDataset and a LengthDataset.
    This allows validation and testing on both or one of
    known structures and de novo samples.
    """

    def __init__(self, pdb_csv=None, samples_cfg=None):
        self.pdb_data = PdbDataset(pdb_csv=pdb_csv) if pdb_csv is not None else None
        self.len_data = LengthDataset(samples_cfg=samples_cfg) if samples_cfg is not None else None
        if self.pdb_data is None and self.len_data is None:
            raise ValueError("CombinedDataset: At least one of pdb_data or len_data must be provided.")
    
    def __len__(self):
        len_pdb = None if self.pdb_data is None else len(self.pdb_data)
        len_len = None if self.len_data is None else len(self.len_data)
        return len_pdb, len_len
    
    def __getitem__(self, idx):
        pdb_idx, gen_idx = idx # list or None, int or None

        chain_feats = self._pdb_dataset[pdb_idx] if pdb_idx is not None else None
        gen_feats = self._length_dataset[gen_idx] if gen_idx is not None else None

        return chain_feats, gen_feats
    
class CombinedDatasetBatchSampler:
                                                                            # TODO: check implementation in case either of the datasets is None
    """
    Data-parallel, distributed batch sampler that groups proteins by length.
    Each batch contains multiple proteins of the same length.

    Requires pre-processed PDB data, using process_pdb_files.py.
    """
    def __init__(self, *, bsampler_cfg, 
                          CombinedDataset, 
                          shuffle=True,
                          num_replicas=None, rank=None,):
        super().__init__()
        self._log = logging.getLogger(__name__)
        self.num_replicas = num_replicas
        self.rank = rank
        if num_replicas is None:
            self.num_replicas = dist.get_world_size()
        if rank is None:
            self.rank = dist.get_rank()
        self.seed = bsampler_cfg.seed
        self.shuffle = shuffle
        self._cfg = bsampler_cfg

        self.cb_data = CombinedDataset
        self.tot_pdbs, self.tot_gens = len(self.cb_data)
        if self.tot_pdbs is not None and self._cfg.num_struc_samples is not None:
            self.num_pdbs = min(self.tot_pdbs, self._cfg.num_struc_samples)
        else:
            self.num_pdbs = self.tot_pdbs

        if self.tot_pdbs is not None:
            self.cb_data.pdb_data.pdb_csv['index'] = list(range(self.tot_pdbs))

        self.epoch = 0
        self.max_batch_size = self._cfg.max_batch_size

        """
        Each replica needs the same number of batches (otherwise leads to bugs).
        This is a constant that depends on num_examples, num_gpus and desired batch_size, 
        where num_examples is max(num_pdbs, num_gens). 
        We (arbitrarily, but sensibly) choose:
        """
        # TODO: begin fixes here


        
        if self.tot_pdbs is None or self.tot_pdbs <= self.tot_gens:
            self._num_batches = math.ceil(self.tot_gens / self.num_replicas)
        elif self.tot_gens is None or self.tot_pdbs > self.tot_gens:
            # self._num_batches = math.ceil( (self.tot_pdbs / self.num_replicas) / self.max_batch_size) * 3
            self._num_batches = self._cfg.num_batches_per_epoch
  
        self._log.info(f'Created dataloader rank {self.rank+1} out of {self.num_replicas}.')
         
    def _replica_epoch_batches(self):
        """
        Returns a list of batches of shape ([pdb_idx1, pdb_idx2,...], gen_idx)
        for the present epoch and replica.
        """
        # ensure all replicas share the same seed in each epoch
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            pdb_idx = torch.randperm(self.tot_pdbs, generator=rng).tolist()
            gen_idx = torch.randperm(self.tot_gens, generator=rng).tolist()
        else:
            pdb_idx = list(range(self.tot_pdbs))
            gen_idx = list(range(self.tot_gens))

        # subsample pdbs if specified
        if self.num_pdbs < self.tot_pdbs:
            pdb_idx = pdb_idx[:self.num_pdbs]

        # split data across processors
        # every num_replicas-th element starting from rank
        replica_pdb_csv = self.cb_data.pdb_data.pdb_csv.iloc[
            pdb_idx[self.rank::self.num_replicas]
        ]
        replica_gens = self.cb_data.len_data._all_sample_ids[
            gen_idx[self.rank::self.num_replicas]
        ]

        # Each structure batch contains multiple proteins of the same length.
        # this minimises padding in each batch, which maximises training efficiency
        sample_order = []
        for seq_len, len_df in replica_pdb_csv.groupby('modeled_seq_len'):
            max_batch_size = min(
                self.max_batch_size,
                self._cfg.max_num_res_squared // seq_len**2 + 1,
            )
            # num batches for this sequence-length
            num_batches = math.ceil(len(len_df) / max_batch_size)
            for i in range(num_batches):
                batch_df = len_df.iloc[i*max_batch_size:(i+1)*max_batch_size]
                batch_indices = batch_df['index'].tolist()
                sample_order.append(batch_indices)
        # mitigate against length bias (NOTE: this retains unequal #batches per length)
        new_order = torch.randperm(len(sample_order), generator=rng).numpy().tolist()
        sample_order = [sample_order[i] for i in new_order]

        return [i for i in itertools.zip_longest(sample_order, replica_gens)]

    def _create_batches(self):
        # Make sure all replicas have the same number of batches. Otherwise leads to bugs.
        # See bugs with shuffling https://github.com/Lightning-AI/lightning/issues/10947
        all_batches = []
        num_augments = -1
        while len(all_batches) < self._num_batches:
            all_batches.extend(self._replica_epoch_batches())
            num_augments += 1
            if num_augments > 1000:
                raise ValueError('Exceeded number of augmentations.')
        if len(all_batches) >= self._num_batches:
            all_batches = all_batches[:self._num_batches]
        self.sample_order = all_batches

    def __iter__(self):
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        return len(self.sample_order)