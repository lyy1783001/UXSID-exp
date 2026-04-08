import sys

sys.path.append('../')
import os
import logging
from fuxictr import datasets
from datetime import datetime
import time
from fuxictr.utils import load_config, set_logger, print_to_json
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import RankDataLoader
import model_zoo
from model_zoo import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/General_config', help='The config directory.')
    parser.add_argument('--expid', type=str, default='CFormer_tmall', help='The model id to run.')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu index, -1 for cpu')

    # overwrite param
    parser.add_argument('--dataset', type=str, default='tmall_900', help='The dataset id to run.')
    parser.add_argument('--embedding_dim', type=str, default='16', help='The embedding size.')
    parser.add_argument('--batch_size', type=str, default='256', help='The batch size.')
    args = vars(parser.parse_args())
    # print(f'args:{args}\n\n')
    # print(f'type(args):{type(args)}\n\n')

    # print args
    print('args:')
    for key in args.keys():
        print(f'{key}: {args[key]}')

    # Load params from config files
    config_dir = args['config']
    experiment_id = args['expid']
    dataset_id = args['dataset']
    print()
    print(f'config_dir: {config_dir}')
    print(f'experiment_id: {experiment_id}')
    print(f'dataset_id: {dataset_id}')
    params = load_config(config_dir, experiment_id, dataset_id)
    params['gpu'] = args['gpu']
    if args['embedding_dim']:
        params['embedding_dim'] = int(args['embedding_dim'])
    if args['batch_size']:
        params['batch_size'] = int(args['batch_size'])

    # set up logger and random seed
    set_logger(params) # 设置logger路径，清空当前logger
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    # # Load feature_map from json
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    print(f"params[data_root]: {params['data_root']}")
    print(f"params[dataset_id]: {params['dataset_id']}")
    print(f"data_dir: {data_dir}")
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    print(f'feature_map_json: {feature_map_json}')
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params) # load feature_map.json
    logging.info("Feature specs: " + print_to_json(feature_map.features))

    # # Get train and validation data generators
    train_gen, valid_gen = RankDataLoader(feature_map,
                                          stage='train',
                                          train_data=params['train_data'],
                                          valid_data=params['valid_data'],
                                          batch_size=params['batch_size'],
                                          data_format=params["data_format"],
                                          shuffle=params['shuffle']).make_iterator()

    # Model initialization and fitting
    model_class = getattr(model_zoo, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters()  # print number of parameters used in model
    # model.load_weights(model.checkpoint)
    model.fit(train_gen, validation_data=valid_gen, epochs=params['epochs'])
    

    # # model.load_weights(model.checkpoint)
    # # logging.info('***** Validation evaluation *****')
    # # model.evaluate(valid_gen)

    # logging.info('***** Test evaluation *****')
    # test_gen = RankDataLoader(feature_map,
    #                           stage='test',
    #                           test_data=params['test_data'],
    #                           batch_size=params['batch_size'],
    #                           data_format=params["data_format"],
    #                           shuffle=False).make_iterator()
    # model.evaluate(test_gen)

