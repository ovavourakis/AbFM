import tree
import math
import random
import logging
import itertools
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import dist
from pytorch_lightning import LightningDataModule

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
        
        # TODO: don't do this conversion here, do it whenever you pre-process the data
        res_idx = torch.tensor(processed_feats['residue_index'])

        # re-index residues starting at 1 (heavy chain) and 1001 (light chain), preserving gaps
        light_chain_start = torch.argmax((res_idx >= 1000).int()).item()
        heavy_chain_res_idx = res_idx[:light_chain_start] - torch.min(res_idx[:light_chain_start]) + 1
        light_chain_res_idx = res_idx[light_chain_start:] - torch.min(res_idx[light_chain_start:]) + 1001
        res_idx = torch.cat([heavy_chain_res_idx, light_chain_res_idx], dim=0)

        return {
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
        chain_feats = self._process_csv_row(csv_row['processed_path'].iloc[0])
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
        # would have to return integer which is not appropriate for this class
        raise NotImplementedError("CombinedDataset: __len__ not implemented.")
    
    def len(self):
        len_pdb = None if self.pdb_data is None else len(self.pdb_data)
        len_len = None if self.len_data is None else len(self.len_data)
        return (len_pdb, len_len)
    
    def __getitem__(self, idx):
        # print('INDEX INSIDE COMBINEDDATASET.__getitem__():', idx) # TODO: remove this
        pdb_idx, gen_idx = idx # list or None, int or None

        chain_feats = self.pdb_data[pdb_idx] if pdb_idx is not None else []
        gen_feats = self.pdb_data[gen_idx] if gen_idx is not None else []

        return chain_feats, gen_feats
    
class CombinedDatasetBatchSampler:
    """
    Individual batches have shape in 
        [([pdb_idx1, pdb_idx2,...], gen_idx)] or
        [([None], gen_idx)] or
        [([pdb_idx1, pdb_idx2,...], None)].
    The outer list is necessary, or PyTorch's DataLoader will unpack
    the tuple and not pass it to Dataset.__getitem__() as a tuple.
    """
    def __init__(self, *, bsampler_cfg, 
                          CombinedDataset, 
                          shuffle=True,
                          num_replicas=None, rank=None):
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
        self.tot_pdbs, self.tot_gens = self.cb_data.len()
        if self.tot_pdbs is not None and self._cfg.num_struc_samples is not None:
            self.num_pdbs = min(self.tot_pdbs, self._cfg.num_struc_samples)
        else:
            self.num_pdbs = self.tot_pdbs

        if self.tot_pdbs is not None:
            self.cb_data.pdb_data.pdb_csv['index'] = list(range(self.tot_pdbs))

        self.epoch = 0
        self.max_batch_size = self._cfg.max_batch_size

        print('TOT_PDBS:', self.tot_pdbs, 'TOT_GENS:', self.tot_gens, 'NUM_PDBS:', self.num_pdbs)

        """
        Each replica needs the same number of batches (otherwise leads to bugs).
        This is a constant that depends on num_examples, num_gpus and desired batch_size, 
        where num_examples is max(num_pdbs, num_gens). 
        We (arbitrarily, but sensibly) choose:
        """
        # both datasets present
        if self.tot_pdbs is not None and self.tot_gens is not None:
            if self.tot_pdbs <= self.tot_gens:
                self._num_batches = math.ceil(self.tot_gens / self.num_replicas)
            else:
                # self._num_batches = math.ceil( (self.tot_pdbs / self.num_replicas) / self.max_batch_size) * 3
                self._num_batches = self._cfg.num_batches_per_epoch_per_gpu
        # just pdbs
        elif self.tot_pdbs is not None and self.tot_gens is None:
            # self._num_batches = math.ceil( (self.tot_pdbs / self.num_replicas) / self.max_batch_size) * 3
            self._num_batches = self._cfg.num_batches_per_epoch_per_gpu
        # just gens
        elif self.tot_pdbs is None and self.tot_gens is not None:
            self._num_batches = math.ceil(self.tot_gens / self.num_replicas)
        # neither
        else:
            raise ValueError(
                "CombinedDatasetBatchSampler: At least one of tot_pdbs or tot_gens must be provided.")
  
        self._log.info(f'Created dataloader rank {self.rank+1} out of {self.num_replicas}.')
         
    def _replica_epoch_batches(self):
        """
        Returns a list of batches for the present epoch and replica. 
        Batches have shape in 
        ([pdb_idx1, pdb_idx2,...], gen_idx) or
        ([None], gen_idx) or
        ([pdb_idx1, pdb_idx2,...], None).
        """
        # ensure all replicas share the same seed in each epoch
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)

        if self.tot_pdbs is not None:
            if self.shuffle:
                pdb_idx = torch.randperm(self.tot_pdbs, generator=rng).tolist()
            else:
                pdb_idx = list(range(self.tot_pdbs))
            if self.num_pdbs < self.tot_pdbs: # also subsample pdbs if specified
                pdb_idx = pdb_idx[:self.num_pdbs]
            # split data across processors
            # every num_replicas-th element starting from rank
            replica_pdb_csv = self.cb_data.pdb_data.pdb_csv.iloc[
                pdb_idx[self.rank::self.num_replicas]
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
        
        if self.tot_gens is not None:
            if self.shuffle:
                gen_idx = torch.randperm(self.tot_gens, generator=rng).tolist()
            else:
                gen_idx = list(range(self.tot_gens))
            replica_gens = self.cb_data.len_data._all_sample_ids[
                gen_idx[self.rank::self.num_replicas]
            ]

        if self.tot_pdbs is None:
            return [[i] for i in itertools.zip_longest([None]*len(replica_gens), replica_gens)]
        elif self.tot_gens is None:
            return [[i] for i in itertools.zip_longest(sample_order, [None]*len(sample_order))]
        else:
            return [[i] for i in itertools.zip_longest(sample_order, replica_gens)]

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
        # print('inside Batcher.__iter__(): SAMPLE ORDER:', self.sample_order) # TODO: remove this
        return iter(self.sample_order)

    def __len__(self):
        return len(self.sample_order)

class DataModule(LightningDataModule):
    
    def __init__(self, data_cfg):
        super().__init__()
        self._log = logging.getLogger(__name__)
        self.data_cfg = data_cfg

    def setup(self, stage: str):
        if self.data_cfg.module.inference_only_mode:
            self.gen_set = CombinedDataset(pdb_csv=None, 
                                           samples_cfg=self.data_cfg.inference.samples)
        else:
            # read structure data
            pdb_csv = pd.read_csv(self.data_cfg.dataset.pdbs.csv_path)
            # apply global filters
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len <= self.data_cfg.dataset.pdbs.max_num_res]
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len >= self.data_cfg.dataset.pdbs.min_num_res]
            if self.data_cfg.dataset.pdbs.subset is not None:
                pdb_csv = pdb_csv.iloc[:int(self.data_cfg.dataset.pdbs.subset)]
            pdb_csv = pdb_csv.sort_values('modeled_seq_len', ascending=False)
            
            # train split
            self.train_set = CombinedDataset(pdb_csv=pdb_csv[pdb_csv.split == 'train'],
                                             samples_cfg=None)
            self._log.info(f'{self.train_set.len()[0]} TRAINING pdbs.')
            # val split
            valid_pdbs, valid_gens = None, None
            if self.data_cfg.dataset.valid.use_pdbs:
                valid_pdbs = pdb_csv[pdb_csv.split == 'valid']
            if self.data_cfg.dataset.valid.generate:
                valid_gens = self.data_cfg.dataset.valid.samples
            self.valid_set = CombinedDataset(pdb_csv=valid_pdbs, 
                                             samples_cfg=valid_gens)
            self._log.info(f'{self.valid_set.len()[0]} VALIDATION pdbs; will sub-sample {self.data_cfg.dataset.valid.bsampler.num_struc_samples} per validation.')
            self._log.info(f'Will also generate {self.valid_set.len()[1]} samples per validation.')
            # test split
            test_pdbs, test_gens = None, None
            if self.data_cfg.dataset.test.use_pdbs:
                test_pdbs = pdb_csv[pdb_csv.split == 'test']
            if self.data_cfg.dataset.test.generate:
                test_gens = self.data_cfg.dataset.test.samples
            self.test_set = CombinedDataset(pdb_csv=test_pdbs,
                                            samples_cfg=test_gens)
            self._log.info(f'{self.test_set.len()[0]} TESTING pdbs.')
            self._log.info(f'Will also generate {self.test_set.len()[1]} samples per test.')

    def get_dataloader(self, type, rank=None, num_replicas=None):
        num_workers = self.data_cfg.module.loaders.num_workers
        prefetch_factor = None if num_workers == 0 else self.data_cfg.module.loaders.prefetch_factor
        
        if type == 'train':
            dataset = self.train_set
            bsampler_cfg = self.data_cfg.dataset.train.bsampler
        elif type == 'valid':
            dataset = self.valid_set
            bsampler_cfg = self.data_cfg.dataset.valid.bsampler
        elif type == 'test':
            dataset = self.test_set
            bsampler_cfg = self.data_cfg.dataset.test.bsampler
        elif type == 'sample':
            dataset = self.gen_set
            bsampler_cfg = self.data_cfg.inference.bsampler
        else:
            raise ValueError(f'Unknown dataloader type {type}.')
        
        bsampler=CombinedDatasetBatchSampler(bsampler_cfg=bsampler_cfg,
                                             CombinedDataset=dataset,
                                             num_replicas=num_replicas,
                                             rank=rank)

        return DataLoader(dataset, batch_sampler=bsampler, 
                                   num_workers=num_workers,
                                   prefetch_factor=prefetch_factor,
                                   pin_memory=False,
                                   persistent_workers=True if num_workers > 0 else False)

    def train_dataloader(self, rank=None, num_replicas=None):
        return self.get_dataloader('train', rank=rank, num_replicas=num_replicas)

    def val_dataloader(self, rank=None, num_replicas=None):
        return self.get_dataloader('valid', rank=rank, num_replicas=num_replicas)

    def test_dataloader(self, rank=None, num_replicas=None):
        return self.get_dataloader('test', rank=rank, num_replicas=num_replicas)

    def predict_dataloader(self, rank=None, num_replicas=None):
        return self.get_dataloader('sample', rank=rank, num_replicas=num_replicas)