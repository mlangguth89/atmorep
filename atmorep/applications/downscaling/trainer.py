####################################################################################################
#
#  Copyright (C) 2022
#
####################################################################################################
#
#  project     : atmorep
#
#  author      : atmorep collaboration
# 
#  description :
#
#  license     :
#
####################################################################################################

import os
import torch
import torchinfo
import numpy as np
import code
# code.interact(local=locals())

import os
import pathlib
import datetime
import time
import math
from typing import TypeVar

import functools

import wandb

from atmorep.core.trainer import Trainer_Base
from atmorep.core.atmorep_model import AtmoRep
from atmorep.core.atmorep_model import AtmoRepData

from atmorep.applications.downscaling.atmorep_downscaling import AtmoRepDownscaling

from atmorep.training.bert import prepare_batch_BERT_multifield
from atmorep.transformer.transformer_base import positional_encoding_harmonic

import atmorep.utils.utils as utils
from atmorep.utils.utils import Config
from atmorep.utils.utils import setup_wandb
from atmorep.utils.utils import shape_to_str
from atmorep.utils.utils import get_model_filename
from atmorep.utils.utils import relMSELoss
from atmorep.utils.utils import init_torch
from atmorep.utils.utils import Gaussian
from atmorep.utils.utils import CRPS
from atmorep.utils.utils import NetMode
from atmorep.utils.utils import tokenize

import atmorep.config.config as config

####################################################################################################
class Trainer_Downscaling( Trainer_Base) :

  ###################################################
  def __init__( self, cf_downscaling, devices) :

    cf = utils.Config()
    cf.load_json( cf_downscaling.base_model_id)
    # overwrite all info that is common and add new one
    for key, value in cf_downscaling.get_self_dict().items() :
      cf.__dict__[key] = value
    # save merged config
    if cf.with_wandb and 0 == cf.hvd_rank :
      cf.write_json( wandb)
      cf.print()

    Trainer_Base.__init__( self, cf, devices)

    p = torch.randint( low=0, high=100000, size=(1,))[0].item()
    self.rngs = [np.random.default_rng(i*p) for i in range( len(cf.fields) * 8  )]
    self.rngs_targets = [np.random.default_rng(i*p) for i in range( len(cf.fields) * 8  )]
    self.pre_batch = functools.partial( prepare_batch_BERT_multifield, self.cf, self.rngs, 
                                                                       self.cf.fields, 'identity' )
    # self.pre_batch_targets = functools.partial( prepare_batch_BERT_multifield, self.cf, 
    #                                           self.rngs_targets, self.cf.fields_targets, 'BERT' )
    self.pre_batch_targets = None

    self.mode_test = False
    self.rng = np.random.default_rng()

    self.save_test_once = True

  ###################################################
  def create( self) :
    
    assert False, 'not implemented, in particular proper initalization of AtmoRep'

  ###################################################
  def load_create( self) :

    net = AtmoRepDownscaling.load_create( self.cf, self.devices)
    self.model = AtmoRepData( net)

    self.model.create( self.pre_batch, self.devices, False, self.pre_batch_targets )

    # overwrite fields predictions with fields_targets
    cf = self.cf

    fields_prediction = []
    self.fields_prediction_idx = []
    self.loss_weights = torch.zeros( len(cf.fields_targets) )
    for ifield, field in enumerate(cf.fields_targets) :
      if field[7][5] > 0. : 
        self.loss_weights[ifield] = field[7][5]
        fields_prediction.append( [field[0], field[7][5] ])
        self.fields_prediction_idx.append( ifield)
    # update 
    cf.fields_prediction = fields_prediction

    # TODO: pass the properly to model / net
    self.model.net.encoder_to_decoder = self.encoder_to_decoder
    self.model.net.decoder_to_tail = self.decoder_to_tail
    self.model.net.decoder_to_downscaler = self.decoder_to_downscaler

    return self

  ###################################################
  @classmethod
  def load( Typename, cf, model_id, epoch, devices) :
    
    trainer = Typename( cf, devices).load_create()
    trainer.model.net.load( model_id, devices, cf, epoch)

    print( 'Loaded model id = {} at epoch = {}.'.format( model_id, epoch) )

    return trainer

  ###################################################
  def prepare_batch( self, xin) :
    '''Move data to device and some additional final preprocessing before model eval'''

    cf = self.cf
    devs = self.devices

    # unpack loader output
    # xin[0] since BERT does not have targets
    (sources, token_infos, _, _, _) = xin[0]

    # network input
    dev = self.device_in
    batch_data_core = [ ( sources[i].to( devs[ cf.fields[i][1][3] ], non_blocking=True), 
        token_infos[i].to( self.devices[0], non_blocking=True) ) for i in range(len(sources)) ]

    # target
    dev = self.device_out
    self.targets = []
    self.targets_token_infos = []
    fields_injected = []
    # for all fields_target
    for ifield, target in enumerate( xin[1] ) :
      tok_size = self.cf.fields_targets[ifield][4]
      temp = []
      temp2 = []

      # process vertical levels
      for target_vl, target_token_info_vl in target :

        # TODO: all/most of this should be moved to pre_batch_targets 
        # item is field data and token_info
        target_vl_tok = tokenize( target_vl, tok_size).unsqueeze(1).to( dev, non_blocking=True)
        shape = [-1] + list(target_vl_tok.shape[1:-3]) + [self.cf.size_token_info]
        target_token_info_vl = target_token_info_vl.reshape( shape)

        # select single time step as downscaling target: currently middle one
        # TODO: should be specifiable parameter
        tstep = 3 if target_vl_tok.shape[2] > 2 else 0   # static fields need tstep = 0
        target_vl_tok = target_vl_tok[:,:,tstep].unsqueeze(2)
        
        temp.append( target_vl_tok )
        temp2.append( target_token_info_vl[:,:,tstep].unsqueeze(2) )

      # merge vertical levels
      target = torch.cat( temp, 1).flatten( -3, -1).flatten( 1, 4)
      target_token_infos = torch.cat( temp2, 1).flatten( 1, -2) 
      # targets: all fields with loss weight > 0.
      if self.cf.fields_targets[ifield][7][5] > 0. :
        self.targets.append( target.flatten( 0, 1))
        self.targets_token_infos.append( target_token_infos)
      # no parent field is specified -> injected (TODO: cleaner mechanism for this)
      if len(self.cf.fields_targets[ifield][7][4]) == 0 :
        fields_injected.append( (target.to(dev).unsqueeze(1).unsqueeze(1), 
                                 target_token_infos.to(dev)) )
    
    # move to model for later injection
    self.model.net.fields_downscaling_injected = fields_injected

    return batch_data_core

  ###################################################
  def encoder_to_decoder( self, embeds_layers) :
    return ([embeds_layers[i][-1] for i in range(len(embeds_layers))] , embeds_layers )

  ###################################################
  def decoder_to_downscaler( self, idx, token_seq) :

    # parent_field = self.cf.fields_targets[idx][6][4]
    # # TODO, handle multiple partent fields
    # found = False
    # for field_info_parent in self.cf.fields : 
    #   if parent_field == field_info_parent[0] :
    #     break
    # if not found :
    #   return (None, None)
    field_info_parent = None
    parent_field = self.cf.fields[idx][0]
    for didx, field_info in enumerate(self.cf.fields_targets) :
      if field_info[7][4] == [parent_field] :
        field_info_parent = self.cf.fields[idx]
        break
    if not field_info_parent :
      return (None, None)

    # recover individual space dimensions
    num_toks_space = field_info_parent[3][1:]
    token_seq = token_seq.reshape( list(token_seq.shape[:3]) + num_toks_space + [-1])

    # select time step (middle one)
    token_seq = token_seq[:,:,6].unsqueeze(2)
    
    # if not self.mode_test :

    #   token_seq_shape = token_seq.shape
    #   token_seq = token_seq.flatten( 0, -2)
      
    #   perm = self.rng.permutation( token_seq.shape[0])
    #   token_seq[ perm[ int( 0.1 * perm.shape[0]) ] , : ] = 0.
      
    #   token_seq = token_seq.reshape( token_seq_shape)

    token_seq = token_seq.flatten( -3, -2)

    dev = self.devices[ self.cf.fields_targets[0][1][3] ]
    return ( token_seq.to( dev, non_blocking=True), 
             self.targets_token_infos[didx].to( dev, non_blocking=True) )
    # return (token_seq, self.targets_token_infos[idx].to( token_seq.device, non_blocking=True))

  ###################################################
  def decoder_to_tail( self, idx_pred, pred) :
    '''Positional encoding of masked tokens for tail network evaluation'''

    field_idx = self.fields_prediction_idx[idx_pred]
    dev = pred.device # self.devices[ self.cf.fields[field_idx][1][3] ]

    pred = pred.flatten( 0, 3)
    num_tokens = self.cf.fields_targets[field_idx][3]
    idx = torch.arange( np.prod( pred.shape[1:-1]), device=dev)

    # compute space time indices of all tokens
    num_tokens_space = num_tokens[1] * num_tokens[2]
    # remove offset introduced by linearization
    target_idxs_t = (idx / num_tokens_space).int()
    temp = torch.remainder( idx, num_tokens_space)
    target_idxs_x = (temp / num_tokens[1]).int()
    target_idxs_y = torch.remainder( temp, num_tokens[2])

    # apply harmonic positional encoding
    dim_embed = pred.shape[-1]
    pe = torch.zeros( target_idxs_x.shape[0], dim_embed, device=dev)
    xs = (2. * np.pi / dim_embed) * torch.arange( 0, dim_embed, 2, device=dev) 
    pe[:, 0::2] = 0.5 * torch.sin( torch.outer( 8 * target_idxs_x, xs) ) \
                    + torch.sin( torch.outer( target_idxs_t, xs) )
    pe[:, 1::2] = 0.5 * torch.cos( torch.outer( 8 * target_idxs_y, xs) ) \
                    + torch.cos( torch.outer( target_idxs_t, xs) )

    pred += pe

    return pred
