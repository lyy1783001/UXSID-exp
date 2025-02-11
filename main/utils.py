import numpy as np
import logging
import os
from torch.utils.data import DataLoader, ConcatDataset
from fuxictr.pytorch.dataloaders.parquet_dataloader import ParquetDataset, BatchCollator
import os
import logging
import numpy as np
import gc
import multiprocessing as mp
import polars as pl
from fuxictr.preprocess.build_dataset import split_train_test, transform, transform_block



class ParquetConcatDataLoader(DataLoader):
    def __init__(self, feature_map, data_path_list: list, batch_size=32, shuffle=False,
                 num_workers=1, **kwargs):
        datasets = []
        for data_path in data_path_list:
            if not data_path.endswith(".parquet"):
                data_path += ".parquet"
            datasets.append(ParquetDataset(feature_map, data_path))
        self.dataset = ConcatDataset(datasets)
        super().__init__(dataset=self.dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers,
                         collate_fn=BatchCollator(feature_map))
        self.num_samples = len(self.dataset)
        self.num_blocks = 1
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

    def __len__(self):
        return self.num_batches
    

class ClusterDataLoader(object):
    def __init__(self, feature_map, stage="both", train_data=None, 
                 batch_size=32, shuffle=True, data_format="npz", **kwargs):
        logging.info("Loading datasets...")

        if kwargs.get("data_loader"):
            DataLoader = kwargs["data_loader"]
        else:
            assert data_format == 'parquet', 'Only support parquet data_format'
            DataLoader = ParquetConcatDataLoader
        self.stage = stage


        train_gen = DataLoader(feature_map, train_data, split="train", batch_size=batch_size,
                                shuffle=shuffle, **kwargs)
        logging.info(
            "Train samples: total/{:d}, blocks/{:d}"
            .format(train_gen.num_samples, train_gen.num_blocks)
        )  

        self.train_gen = train_gen

    def make_iterator(self):
        logging.info("Loading data done.")
        return self.train_gen
    

def set_logger(params):
    dataset_id = params['dataset_id']
    model_id = params.get('model_id', '')
    log_dir = os.path.join(params.get('model_root', './checkpoints'), dataset_id)
    os.makedirs(log_dir, exist_ok=True)

    n_cluster = params.get('n_cluster', '*')
    learning_rate = params.get('learning_rate', '*')
    embedding_regularizer = params.get('embedding_regularizer', '*')
    d_model = params.get('d_model', '*')
    nhead = params.get('nhead', '*')
    num_encoder_layers = params.get('num_encoder_layers', '*')
    num_decoder_layers = params.get('num_decoder_layers', '*')
    dim_feedforward = params.get('dim_feedforward', '*')
    dropout = params.get('dropout', '*')

    recon_loss = params.get('recon_loss', False)
    recon_weight = params.get('recon_weight', 1.0)
    cl_loss = params.get('cl_loss', False)
    cl_weight = params.get('cl_weight', 1.0)
    orth_loss = params.get('orth_loss', False)
    orth_weight = params.get('orth_weight', 1.0)
    bce_loss = params.get('bce_loss', False)
    bce_weight = params.get('bce_weight', 1.0)
    kl_loss = params.get('kl_loss', False)
    kl_weight = params.get('kl_weight', 1.0)
    js_loss = params.get('js_loss', False)
    js_weight = params.get('js_weight', 1.0)
    bl_loss = params.get('bl_loss', False)
    bl_weight = params.get('bl_weight', 1.0)
    temperature = params.get('temperature', 0.01)

    use_weight = params.get('use_weight', '*')

    basename = f"dropout{dropout}_" + \
                f"cluster{n_cluster}_" + \
                f"{use_weight}_" + \
                f"lr{learning_rate}_" + \
                f"reg{embedding_regularizer}_" + \
                f"dmodel{d_model}_" + \
                f"nhead{nhead}_" + \
                f"enc{num_encoder_layers}_" + \
                f"dec{num_decoder_layers}_" + \
                f"ffn{dim_feedforward}_"
    if recon_loss and recon_weight > 0:
        basename += f"{recon_weight}recon_loss_"
    if cl_loss and cl_weight > 0:
        basename += f"{cl_weight}cl_loss_t{temperature}_"
    if orth_loss and orth_weight > 0:
        basename += f"{orth_weight}orth_loss_"
    if bce_loss and bce_weight > 0:
        basename += f"{bce_weight}bce_loss_"
    if kl_loss and kl_weight > 0:
        basename += f"{kl_weight}kl_loss_"
    if js_loss and js_weight > 0:
        basename += f"{js_weight}js_loss_"
    if bl_loss and bl_weight > 0:
        basename += f"{bl_weight}bl_loss_"

    log_file = os.path.join(log_dir, basename + '.log')

    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]: 
        logging.root.removeHandler(handler)
        
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s P%(process)d %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(log_file, mode='w'),
                                  logging.StreamHandler()])
    

def build_dataset(feature_encoder, train_data=None, valid_data=None, test_data=None,
                  valid_size=0, test_size=0, split_type="sequential", data_block_size=0,
                  rebuild_dataset=True, **kwargs):
    """ Build feature_map and transform data """
    if rebuild_dataset:
        feature_map_path = os.path.join(feature_encoder.data_dir, "feature_map.json")
        if os.path.exists(feature_map_path):
            logging.warn(f"Skip rebuilding {feature_map_path}. " 
                + "Please delete it manually if rebuilding is required.")

        # Load data files
        train_ddf = feature_encoder.read_data(train_data, **kwargs)
        valid_ddf = None
        test_ddf = None

        # Split data for train/validation/test
        if valid_size > 0 or test_size > 0:
            valid_ddf = feature_encoder.read_data(valid_data, **kwargs)
            test_ddf = feature_encoder.read_data(test_data, **kwargs)
            # TODO: check split_train_test in lazy mode
            train_ddf, valid_ddf, test_ddf = split_train_test(train_ddf, valid_ddf, test_ddf, 
                                                              valid_size, test_size, split_type)
        
        # fit and transform train_ddf
        train_ddf = feature_encoder.preprocess(train_ddf)
        # feature_encoder.fit(train_ddf, rebuild_dataset=True, **kwargs)
        # del train_ddf
        # gc.collect()

        # Transfrom valid_ddf
        if valid_ddf is None and (valid_data is not None):
            valid_ddf = feature_encoder.read_data(valid_data, **kwargs)
        if valid_ddf is not None:
            valid_ddf = feature_encoder.preprocess(valid_ddf)
            # del valid_ddf
            # gc.collect()

        # Transfrom test_ddf
        if test_ddf is None and (test_data is not None):
            test_ddf = feature_encoder.read_data(test_data, **kwargs)
        if test_ddf is not None:
            test_ddf = feature_encoder.preprocess(test_ddf)
            # del test_ddf
            # gc.collect()

        all_ddf = [train_ddf]
        if valid_ddf is not None:
            all_ddf.append(valid_ddf)
        if test_ddf is not None:
            all_ddf.append(test_ddf)

        all_ddf = pl.concat(all_ddf, how='vertical')
        feature_encoder.fit(all_ddf, rebuild_dataset=True, **kwargs)
        del all_ddf
        gc.collect()

        transform(feature_encoder, train_ddf, 'train', block_size=data_block_size)
        del train_ddf
        gc.collect()
        if valid_ddf is not None:
            transform(feature_encoder, valid_ddf, 'valid', block_size=data_block_size)
            del valid_ddf
            gc.collect()
        if test_ddf is not None:
            transform(feature_encoder, test_ddf, 'test', block_size=data_block_size)
            del test_ddf
            gc.collect()

        logging.info("Transform csv data to parquet done.")

        train_data, valid_data, test_data = (
            os.path.join(feature_encoder.data_dir, "train"), \
            os.path.join(feature_encoder.data_dir, "valid"), \
            os.path.join(feature_encoder.data_dir, "test") if (
                test_data or test_size > 0) else None
        )
    
    else: # skip rebuilding data but only compute feature_map.json
        feature_encoder.fit(train_ddf=None, rebuild_dataset=False, **kwargs)
    
    # Return processed data splits
    return train_data, valid_data, test_data