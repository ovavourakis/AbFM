import tree
import math
import random
import logging
import itertools
import numpy as np
import pandas as pd

from typing import TypeVar, Optional, Iterator
T_co = TypeVar('T_co', covariant=True)

import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler
from torch.utils.data.distributed import dist, DistributedSampler

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.combined_loader import CombinedLoader

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
        
        # re-index residues starting at 1 (VH) and jumping by 50 at start of VL, preserving gaps
        # (VL indices currently offset by 1000 from process_ab_pdb_files)
        # TODO: don't do this conversion here, do it whenever you pre-process the data
        res_idx = torch.tensor(processed_feats['residue_index'])
        light_chain_start = torch.argmax((res_idx >= 1000).int()).item()
        heavy_chain_res_idx = res_idx[:light_chain_start] - torch.min(res_idx[:light_chain_start]) + 1
        heavy_chain_end = heavy_chain_res_idx[-1].item()

        light_chain_offset = 50
        light_chain_res_idx = res_idx[light_chain_start:] - torch.min(res_idx[light_chain_start:]) + (heavy_chain_end + light_chain_offset)
        res_idx = torch.cat([heavy_chain_res_idx, light_chain_res_idx], dim=0)

        return {
            'file': processed_file_path,
            'aatype': chain_feats['aatype'],
            'res_idx': res_idx, # array; starting at 1 (VH) and 1001 (VL), preserving gaps
            'rotmats_1': rotmats_1,
            'trans_1': trans_1,
            'res_mask': torch.tensor(processed_feats['bb_mask']).int(),
        }

    def __len__(self):
        return len(self.pdb_csv)

    def __getitem__(self, idx):
        # return one structure from pdb csv
        csv_row = self.pdb_csv.iloc[idx]
        chain_feats = self._process_csv_row(csv_row['processed_path'])#.iloc[0])
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

class DistributedPdbBatchSampler(DistributedSampler):

    def __init__(self, dataset: Dataset,
                       bsampler_cfg=None,
                       num_replicas: Optional[int] = None,
                       rank: Optional[int] = None, 
                       shuffle: bool = True,
                       seed: int = 0, 
                       drop_last: bool = False
        ) -> None:
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle,
                         seed=seed, drop_last=drop_last)
        self._log = logging.getLogger(__name__)
        
        self.cfg = bsampler_cfg
        self.batch_size = self.cfg.batch_size

        self.make_batches_for_epoch_and_replica() # creates self.batches = [[idx1, idx2, ...], [idxN, idxM, ...], ...]
        self._log.info(f'Created dataloader rank {self.rank+1} out of {self.num_replicas}.\n\
                         Contains {len(self.batches)} batches of size {self.batch_size}.')

    def make_batches_for_epoch_and_replica(self):
        # start from complete dataset (train, val or test)
        full_dataset = self.dataset.pdb_csv
        full_dataset['index'] = list(range(len(full_dataset))) # needed later
        
        # subsample one pdb per sequence-similarity cluster (for this epoch)
        seed = self.seed + self.epoch # unique to each epoch, common across replicas
        full_dataset = full_dataset.groupby('cluster_ids').sample(n=1, random_state=seed).reset_index(drop=True)

        # shuffle pdb indices, if specified
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(seed)
            # indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
            indices = torch.randperm(len(full_dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            # indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
            indices = list(range(len(full_dataset)))  # type: ignore[arg-type]
        # also subsample pdb indices if specified
        if self.cfg.num_struc_samples is not None:
            indices = indices[:self.cfg.num_struc_samples]

        # apply shuffling and subsampling
        shuffled_dataset = full_dataset.iloc[indices]

        # create length-homogenous batches of equal size batch_size
        batches = []
        for seq_len, len_df in shuffled_dataset.groupby('modeled_seq_len'):
            
            # num of *complete* batches for *this* sequence-length
            # this will drop some structures in the incomplete last batch 
            # (for this epoch)
            num_batches = math.floor(len(len_df) / self.batch_size)

            for i in range(num_batches):
                batch_df = len_df.iloc[i*self.batch_size:(i+1)*self.batch_size]
                batch_indices = batch_df['index'].tolist()
                batches.append(batch_indices)

        # shuffle batches to mix lengths
        new_order = torch.randperm(len(batches), generator=g).tolist()
        batches = [batches[i] for i in new_order]
        total_num_batches = len(batches)
        assert total_num_batches > 0, 'No batches created.'
        
        # make sure the number of batches is divisible by num_replicas
        num_augments = -1
        while total_num_batches % self.num_replicas != 0:
            if not self.drop_last:
                total_num_batches += 1
                num_augments += 1
            else:
                total_num_batches -= 1
                num_augments += 1
            if num_augments > 1000:
                raise ValueError('Exceeded number of augmentations.')
        if total_num_batches >= len(batches):
            padding_size = total_num_batches - len(batches)
            batches += batches[:padding_size]
        else:
            batches = batches[:total_num_batches]
        assert len(batches) == total_num_batches, f'len(batches)={len(batches)} != total_num_batches={total_num_batches}'
        
        # distribute batches across replicas by subsampling
        # every nth batch for current replica (and epoch)
        replica_batches = batches[self.rank::self.num_replicas]  
        self.batches = replica_batches
        assert len(self.batches) > 0, 'No batches created.'
        for batch in self.batches:
            assert len(batch) == self.batch_size, f"Batch length is {len(batch)}, but expected length is {self.batch_size}."

    def __iter__(self) -> Iterator[T_co]:
        if self.epoch > 0:
            self.make_batches_for_epoch_and_replica()
        self.epoch += 1
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)

class DataModule(LightningDataModule):
    
    def __init__(self, data_cfg):
        super().__init__()
        self._log = logging.getLogger(__name__)
        self.data_cfg = data_cfg

    def setup(self, stage: str):
        if self.data_cfg.module.inference_only_mode:
            self.gen_set = LengthDataset(samples_cfg=self.data_cfg.inference.samples)
        else:
            # preliminaries for structure data
            pdb_csv = pd.read_csv(self.data_cfg.dataset.pdbs.csv_path)
            # apply global filters
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len <= self.data_cfg.dataset.pdbs.max_num_res]
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len >= self.data_cfg.dataset.pdbs.min_num_res]
            if self.data_cfg.dataset.pdbs.subset is not None:
                pdb_csv = pdb_csv.iloc[:int(self.data_cfg.dataset.pdbs.subset)]
            
            # train split
            # self.train_pdbs = PdbDataset(pdb_csv=pdb_csv[pdb_csv.split == 'train'])
            self.train_pdbs = PdbDataset(pdb_csv=pdb_csv)
            self._log.info(f'{len(self.train_pdbs)} TRAINING pdbs.')
            # val split
            self.valid_pdbs, self.valid_gens = None, None
            if self.data_cfg.dataset.valid.use_pdbs:
                valid_pdbs = pdb_csv
                # valid_pdbs = pdb_csv[pdb_csv.split == 'valid']
                self.valid_pdbs = PdbDataset(pdb_csv=valid_pdbs)
                self._log.info(f'{len(self.valid_pdbs)} VALIDATION pdbs; will sub-sample {self.data_cfg.dataset.valid.bsampler.num_struc_samples} per validation run.')
            if self.data_cfg.dataset.valid.generate:
                valid_gens = self.data_cfg.dataset.valid.samples
                self.valid_gens = LengthDataset(samples_cfg=valid_gens)
                self._log.info(f'Will also generate {len(self.valid_gens)} samples per validation run.')
            # test split
            self.test_pdbs, self.test_gens = None, None
            if self.data_cfg.dataset.test.use_pdbs:
                test_pdbs = pdb_csv
                # test_pdbs = pdb_csv[pdb_csv.split == 'test']
                self.test_pdbs = PdbDataset(pdb_csv=test_pdbs)
                self._log.info(f'{len(self.test_pdbs)} TESTING pdbs; will sub-sample {self.data_cfg.dataset.test.bsampler.num_struc_samples} per test run.')
            if self.data_cfg.dataset.test.generate:
                test_gens = self.data_cfg.dataset.test.samples
                self.test_gens = LengthDataset(samples_cfg=test_gens)
                self._log.info(f'Will also generate {len(self.test_gens)} samples per test run.')

    def get_dataloader(self, type, rank=None, num_replicas=None):
        
        num_workers = self.data_cfg.module.loaders.num_workers
        prefetch_factor = None if num_workers == 0 else self.data_cfg.module.loaders.prefetch_factor
        
        if type == 'train':
            bsampler_cfg = self.data_cfg.dataset.train.bsampler
            pdbs = self.train_pdbs
            gens = None
        elif type == 'valid':
            bsampler_cfg = self.data_cfg.dataset.valid.bsampler
            pdbs = self.valid_pdbs
            gens = self.valid_gens
        elif type == 'test':
            bsampler_cfg = self.data_cfg.dataset.test.bsampler
            pdbs = self.test_pdbs
            gens = self.test_gens
        elif type == 'sample':
            pdbs = None
            gens = self.gen_set
        else:
            raise ValueError(f'Unknown dataloader type {type}.')

        if pdbs is not None:
            bsampler = DistributedPdbBatchSampler(dataset=pdbs,
                                                bsampler_cfg=bsampler_cfg,
                                                num_replicas=num_replicas,
                                                rank=rank)
            pdb_loader = DataLoader(pdbs,
                                    batch_sampler=bsampler,
                                    num_workers=num_workers,
                                    prefetch_factor=prefetch_factor,
                                    pin_memory=False,
                                    persistent_workers=True if num_workers > 0 else False)
        if gens is not None:
            gen_loader = DataLoader(gens,
                                    batch_size=1,
                                    num_workers=num_workers,
                                    prefetch_factor=prefetch_factor,
                                    pin_memory=False,
                                    persistent_workers=True if num_workers > 0 else False)

        # both types present
        if pdbs is not None and gens is not None:
            iterables = {'struc' : pdb_loader, 'gen' : gen_loader}
            print('Both structures and generation paramters passed. Starting with CombinedLoader.')
            return CombinedLoader(iterables, 'max_size')
        # just generation parameters
        elif pdbs is None and gens is not None:
            return gen_loader
        # just pdbs
        elif pdbs is not None and gens is None:
            return pdb_loader
        # neither
        else:
            raise ValueError('No data present for validation or test dataloader.')

    def train_dataloader(self, rank=None, num_replicas=None):
        return self.get_dataloader('train', rank=rank, num_replicas=num_replicas)

    def val_dataloader(self, rank=None, num_replicas=None):
        return self.get_dataloader('valid', rank=rank, num_replicas=num_replicas)

    def test_dataloader(self, rank=None, num_replicas=None):
        return self.get_dataloader('test', rank=rank, num_replicas=num_replicas)

    def predict_dataloader(self, rank=None, num_replicas=None):
        return self.get_dataloader('sample', rank=rank, num_replicas=num_replicas)