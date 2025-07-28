from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import functools

from guided_diffusion.models.fp16_util import convert_module_to_f16, convert_module_to_f32
from guided_diffusion.models.nn import (
	checkpoint,
	conv_nd,
	linear,
	avg_pool_nd,
	zero_module,
	normalization,
	timestep_embedding,
)
from guided_diffusion.models.unet_blocks import AttentionPool2d, TimestepBlock, TimestepEmbedSequential 
from guided_diffusion.models.unet_blocks import Upsample, Downsample, ResBlock, AttentionBlock, count_flops_attn
from guided_diffusion.models.unet_blocks import QKVAttentionLegacy, QKVAttention




class EncoderUNetModel(nn.Module):
	"""
	The half UNet model with attention and timestep embedding.

	For usage, see UNet.
	"""

	def __init__(
		self,
		image_size,
		in_channels,
		model_channels,
		num_res_blocks,
		attention_resolutions,
		upscale_factor = None,
		dropout=0,
		channel_mult=(1, 2, 4, 8),
		conv_resample=True,
		dims=2,
		use_checkpoint=False,
		use_fp16=False,
		num_heads=1,
		num_head_channels=-1,
		num_heads_upsample=-1,
		use_scale_shift_norm=False,
		resblock_updown=False,
		use_new_attention_order=False,
	):
		super().__init__()

		if num_heads_upsample == -1:
			num_heads_upsample = num_heads

		self.in_channels = in_channels
		self.model_channels = model_channels
		self.num_res_blocks = num_res_blocks
		self.attention_resolutions = attention_resolutions
		self.dropout = dropout
		self.channel_mult = channel_mult
		self.conv_resample = conv_resample
		self.use_checkpoint = use_checkpoint
		self.dtype = th.float16 if use_fp16 else th.float32
		self.num_heads = num_heads
		self.num_head_channels = num_head_channels
		self.num_heads_upsample = num_heads_upsample

		time_embed_dim = model_channels * 4
		self.time_embed = nn.Sequential(
			linear(model_channels, time_embed_dim),
			nn.SiLU(),
			linear(time_embed_dim, time_embed_dim),
		)

		ch = int(channel_mult[0] * model_channels)
		self.input_blocks = nn.ModuleList(
			[TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
		)
		self.upscale_factor = upscale_factor
		if not upscale_factor is None:
			self.upscale_layer = nn.Upsample(scale_factor = upscale_factor, mode='bilinear')


		self._feature_size = ch
		self.input_block_chans = [ch]
		ds = 1
		for level, mult in enumerate(channel_mult):
			for _ in range(num_res_blocks):
				layers = [
					ResBlock(
						ch,
						time_embed_dim,
						dropout,
						out_channels=int(mult * model_channels),
						dims=dims,
						use_checkpoint=use_checkpoint,
						use_scale_shift_norm=use_scale_shift_norm,
					)
				]
				ch = int(mult * model_channels)
				if ds in attention_resolutions:
					layers.append(
						AttentionBlock(
							ch,
							use_checkpoint=use_checkpoint,
							num_heads=num_heads,
							num_head_channels=num_head_channels,
							use_new_attention_order=use_new_attention_order,
						)
					)
				self.input_blocks.append(TimestepEmbedSequential(*layers))
				self._feature_size += ch
				self.input_block_chans.append(ch)
			if level != len(channel_mult) - 1:
				out_ch = ch
				self.input_blocks.append(
					TimestepEmbedSequential(
						ResBlock(
							ch,
							time_embed_dim,
							dropout,
							out_channels=out_ch,
							dims=dims,
							use_checkpoint=use_checkpoint,
							use_scale_shift_norm=use_scale_shift_norm,
							down=True,
						)
						if resblock_updown
						else Downsample(
							ch, conv_resample, dims=dims, out_channels=out_ch
						)
					)
				)
				ch = out_ch
				self.input_block_chans.append(ch)
				ds *= 2
				self._feature_size += ch

		self.input_block_chans = np.asarray(self.input_block_chans)
	def convert_to_fp16(self):
		"""
		Convert the torso of the model to float16.
		"""
		self.input_blocks.apply(convert_module_to_f16)

	def convert_to_fp32(self):
		"""
		Convert the torso of the model to float32.
		"""
		self.input_blocks.apply(convert_module_to_f32)

	def forward(self, x, timesteps):
		"""
		Apply the model to an input batch.

		:param x: an [N x C x ...] Tensor of inputs.
		:param timesteps: a 1-D batch of timesteps.
		:return: an [N x K] Tensor of outputs.
		"""
		emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
		hs = []
		if not self.upscale_factor is None:
			h = self.upscale_layer(x.type(self.dtype))
		else:
			h = x.type(self.dtype)
		
		for module in self.input_blocks:
			h = module(h, emb)
			hs.append(h)
		
		return hs, emb


class MiddleUNetModel(nn.Module):
	"""
	The half UNet model with attention and timestep embedding.

	For usage, see UNet.
	"""

	def __init__(
		self,
		channels,
		time_embed_dim,
		dropout=0,
		dims=2,
		use_checkpoint=False,
		use_fp16=False,
		num_heads=1,
		num_head_channels=-1,
		use_scale_shift_norm=False,
		use_new_attention_order=False,
	):
		super().__init__()

		self.middle_block = TimestepEmbedSequential(
			ResBlock(
				channels,
				time_embed_dim,
				dropout,
				dims=dims,
				use_checkpoint=use_checkpoint,
				use_scale_shift_norm=use_scale_shift_norm,
			),
			AttentionBlock(
				channels,
				use_checkpoint=use_checkpoint,
				num_heads=num_heads,
				num_head_channels=num_head_channels,
				use_new_attention_order=use_new_attention_order,
			),
			ResBlock(
				channels,
				time_embed_dim,
				dropout,
				dims=dims,
				use_checkpoint=use_checkpoint,
				use_scale_shift_norm=use_scale_shift_norm,
			),
		)

		

	def convert_to_fp16(self):
		"""
		Convert the torso of the model to float16.
		"""
		self.middle_block.apply(convert_module_to_f16)

	def convert_to_fp32(self):
		"""
		Convert the torso of the model to float32.
		"""
		self.middle_block.apply(convert_module_to_f32)

	def forward(self, h, emb):
		"""
		Apply the model to an input batch.

		:param x: an [N x C x ...] Tensor of inputs.
		:param timesteps: a 1-D batch of timesteps.
		:return: an [N x K] Tensor of outputs.
		"""
		h = self.middle_block(h, emb)
		
		return h


class DecoderUNetModel(nn.Module):
	"""
	The half UNet model with attention and timestep embedding.

	For usage, see UNet.
	"""

	def __init__(
		self,
		model_channels,
		channel_mult,
		input_block_chans,
		out_channels,
		out_size = None,
		num_res_blocks=1,
		attention_resolutions=[16],
		dropout=0,
		conv_resample=True,
		dims=2,
		use_checkpoint=False,
		use_fp16=False,
		num_heads=1,
		num_head_channels=-1,
		num_heads_upsample=-1,
		use_scale_shift_norm=False,
		resblock_updown=False,
		use_new_attention_order=False,
	):
		super().__init__()

		if num_heads_upsample == -1:
			num_heads_upsample = num_heads

		self.model_channels = model_channels
		self.num_res_blocks = num_res_blocks
		self.attention_resolutions = attention_resolutions
		self.dropout = dropout
		self.channel_mult = channel_mult
		self.conv_resample = conv_resample
		self.use_checkpoint = use_checkpoint
		self.dtype = th.float16 if use_fp16 else th.float32
		self.num_heads = num_heads
		self.num_head_channels = num_head_channels
		self.num_heads_upsample = num_heads_upsample

		time_embed_dim = 4*model_channels
		ch = int(model_channels*channel_mult[-1])
		ds = 2**(len(channel_mult)-1)
		input_ch = int(model_channels*channel_mult[0])
		self.output_blocks = nn.ModuleList([])
		for level, mult in list(enumerate(channel_mult))[::-1]:
			for i in range(num_res_blocks + 1):
				ich = input_block_chans.pop()
				layers = [
					ResBlock(
						ch + ich,
						time_embed_dim,
						dropout,
						out_channels=int(model_channels * mult),
						dims=dims,
						use_checkpoint=use_checkpoint,
						use_scale_shift_norm=use_scale_shift_norm,
					)
				]
				ch = int(model_channels * mult)
				if ds in attention_resolutions:
					layers.append(
						AttentionBlock(
							ch,
							use_checkpoint=use_checkpoint,
							num_heads=num_heads_upsample,
							num_head_channels=num_head_channels,
							use_new_attention_order=use_new_attention_order,
						)
					)
				if level and i == num_res_blocks:
					out_ch = ch
					layers.append(
						ResBlock(
							ch,
							time_embed_dim,
							dropout,
							out_channels=out_ch,
							dims=dims,
							use_checkpoint=use_checkpoint,
							use_scale_shift_norm=use_scale_shift_norm,
							up=True,
						)
						if resblock_updown
						else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
					)
					ds //= 2
				self.output_blocks.append(TimestepEmbedSequential(*layers))

		if out_size is None:
			self.out = nn.Sequential(
				normalization(ch),
				nn.SiLU(),
				zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
			)
		else:
			self.out = nn.Sequential(
				normalization(ch),
				nn.SiLU(),
				zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
				nn.AdaptiveAvgPool2d(out_size)
			)


	def convert_to_fp16(self):
		"""
		Convert the torso of the model to float16.
		"""
		self.output_blocks.apply(convert_module_to_f16)

	def convert_to_fp32(self):
		"""
		Convert the torso of the model to float32.
		"""
		self.output_blocks.apply(convert_module_to_f32)

	def forward(self, h, hs, emb):
		"""
		Apply the model to an input batch.

		:param x: an [N x C x ...] Tensor of inputs.
		:param timesteps: a 1-D batch of timesteps.
		:return: an [N x K] Tensor of outputs.
		"""
		for module in self.output_blocks:
			h = th.cat([h, hs.pop()], dim=1)
			h = module(h, emb)
		

		return self.out(h)

class KernelUNet(nn.Module):
	"""
	The full UNet model with 2 encoder branches - kernel 64 X 64 and image 256 X 256
	aims to sample from distribution p(h|y)
	"""

	def __init__(
		self,
		kernel_size = 64,
		image_size = 256,
		in_channels = 3,
		model_channels_im = 32,
		model_channels_h = 32,
		num_res_blocks = 1,
		out_channels = 1,
		attention_resolutions = [16],
		dropout=0,
		channel_mult=(1, 2, 4, 8),
		conv_resample=True,
		dims=2,
		num_classes=None,
		use_checkpoint=False,
		use_fp16=False,
		num_heads=1,
		num_head_channels=-1,
		num_heads_upsample=-1,
		use_scale_shift_norm=False,
		resblock_updown=False,
		use_new_attention_order=False,
	):
		super().__init__()

		if num_heads_upsample == -1:
			num_heads_upsample = num_heads

		self.kernel_size = kernel_size
		self.image_size = image_size
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.num_res_blocks = num_res_blocks
		self.attention_resolutions = attention_resolutions
		self.dropout = dropout
		self.channel_mult = channel_mult
		self.conv_resample = conv_resample
		self.num_classes = num_classes
		self.use_checkpoint = use_checkpoint
		self.dtype = th.float16 if use_fp16 else th.float32
		self.num_heads = num_heads
		self.num_head_channels = num_head_channels
		self.num_heads_upsample = num_heads_upsample

		# Some parameters required for GuasianDiffusion, but not relevant for this architecture
		self.channels = out_channels
		self.self_condition = False
		self.random_or_learned_sinusoidal_cond = False


		common_kwargs = {
		'channel_mult': channel_mult,
		'num_res_blocks': num_res_blocks,
		'attention_resolutions': attention_resolutions,
		'dropout': dropout,
		'conv_resample':conv_resample,
		'dims': dims,
		'use_checkpoint': use_checkpoint,
		'use_fp16': use_fp16,
		'num_heads': num_heads,
		'num_head_channels': num_head_channels,
		'num_heads_upsample': num_head_channels,
		'use_scale_shift_norm': use_scale_shift_norm,
		'resblock_updown': resblock_updown,
		'use_new_attention_order': use_new_attention_order}
		
		out_ch_im = int(model_channels_im*channel_mult[-1])
		out_ch_h = int(model_channels_h*channel_mult[-1])
		middle_input_channels = out_ch_im + out_ch_h
		total_model_channels = model_channels_im + model_channels_h
		time_embed_dim_mid = 4*total_model_channels
		# Precompute the channels of each stage of the unet
		ch = int(channel_mult[0] * model_channels_im) +  int(channel_mult[0] * model_channels_h) 
		input_block_chans = [ch]
		for level, mult in enumerate(channel_mult):
			ch = int(mult * model_channels_im) +  int(mult * model_channels_h) 
			input_block_chans.append(ch)
			if level != len(channel_mult) -1:
				input_block_chans.append(ch)

		# Two Encoder-Heads - One for Kernel and Image 
		self.encoder_kernel = EncoderUNetModel(image_size = kernel_size, in_channels = 1, 
			model_channels = model_channels_h, upscale_factor = int(image_size//kernel_size), **common_kwargs)
		self.encoder_image = EncoderUNetModel(image_size = image_size, in_channels = in_channels, 
			model_channels = model_channels_im, upscale_factor = None, **common_kwargs)
		self.middle_unet = MiddleUNetModel(channels = middle_input_channels, time_embed_dim = time_embed_dim_mid)	
		self.decoder = DecoderUNetModel( channel_mult = channel_mult, model_channels = total_model_channels,
		input_block_chans = input_block_chans, out_channels = out_channels, out_size = (kernel_size,kernel_size) )

	def convert_to_fp16(self):
		"""
		Convert the torso of the model to float16.
		"""
		self.encoder_kernel.apply(convert_module_to_f16)
		self.encoder_image.apply(convert_module_to_f16)

	def convert_to_fp32(self):
		"""
		Convert the torso of the model to float32.
		"""
		self.encoder_kernel.apply(convert_module_to_f32)
		self.encoder_image.apply(convert_module_to_f32)
		
	def forward(self, h, y, timesteps, *args, **kwargs):
		"""
		Apply the model to an input batch.

		:param h: an [N x C x ...] Tensor of kernel inputs.
		:param y: an [N] Tensor of blurred images.
		:param timesteps: a 1-D batch of timesteps.
		:return: an [N x C x ...] Tensor of outputs.
		"""
		h_feat, h_emb = self.encoder_kernel(h, timesteps)
		im_feat, im_emb = self.encoder_image(y, timesteps)

		h_mid, emb_mid = th.cat((h_feat[-1],im_feat[-1]),dim=1), th.cat((h_emb,im_emb), dim=1)
		h_mid = self.middle_unet(h_mid, emb_mid)
		concat_features = []
		for h, im in zip(h_feat, im_feat):
			concat_features.append(th.cat((h,im), dim=1))
		h_out = self.decoder(h_mid, concat_features, emb_mid)
			 
		return h_out

		