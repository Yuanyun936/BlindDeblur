"""
Code from Blind-DPS implementation: https://github.com/BlindDPS/blind-dps
"""

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
NUM_CLASSES = 1000

def create_model(
	image_size,
	num_channels,
	num_res_blocks,
	channel_mult="",
	learn_sigma=False,
	class_cond=False,
	use_checkpoint=False,
	attention_resolutions="16",
	num_heads=1,
	num_head_channels=-1,
	num_heads_upsample=-1,
	use_scale_shift_norm=False,
	dropout=0,
	resblock_updown=False,
	use_fp16=False,
	use_new_attention_order=False,
	grayscale=False,
	twochan=False,
	model_path='',
):

	if grayscale:
		in_channels = 1
		out_channels= 1 if not learn_sigma else 2
	elif twochan:
		in_channels = 2
		out_channels = 2 if not learn_sigma else 4
	else:
		in_channels = 3
		out_channels = 3 if not learn_sigma else 6

	if channel_mult == "":
		if image_size == 512:
			channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
		elif image_size == 256:
			channel_mult = (1, 1, 2, 2, 4, 4)
		elif image_size == 128:
			channel_mult = (1, 1, 2, 3, 4)
		elif image_size == 64:
			channel_mult = (1, 2, 3, 4)
		else:
			raise ValueError(f"unsupported image size: {image_size}")
	else:
		channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

	attention_ds = []
	if isinstance(attention_resolutions, int):
		attention_ds.append(image_size // attention_resolutions)
	elif isinstance(attention_resolutions, str):
		for res in attention_resolutions.split(","):
			attention_ds.append(image_size // int(res))
	else:
		raise NotImplementedError

	model= UNetModel(
		image_size=image_size,
		in_channels=in_channels,
		model_channels=num_channels,
		out_channels=out_channels,
		num_res_blocks=num_res_blocks,
		attention_resolutions=tuple(attention_ds),
		dropout=dropout,
		channel_mult=channel_mult,
		num_classes=(NUM_CLASSES if class_cond else None),
		use_checkpoint=use_checkpoint,
		use_fp16=use_fp16,
		num_heads=num_heads,
		num_head_channels=num_head_channels,
		num_heads_upsample=num_heads_upsample,
		use_scale_shift_norm=use_scale_shift_norm,
		resblock_updown=resblock_updown,
		use_new_attention_order=use_new_attention_order,
	)

	model.load_state_dict(th.load(model_path, map_location='cpu'), strict=True)
	return model


class UNetModel(nn.Module):
	"""
	The full UNet model with attention and timestep embedding.

	:param in_channels: channels in the input Tensor.
	:param model_channels: base channel count for the model.
	:param out_channels: channels in the output Tensor.
	:param num_res_blocks: number of residual blocks per downsample.
	:param attention_resolutions: a collection of downsample rates at which
		attention will take place. May be a set, list, or tuple.
		For example, if this contains 4, then at 4x downsampling, attention
		will be used.
	:param dropout: the dropout probability.
	:param channel_mult: channel multiplier for each level of the UNet.
	:param conv_resample: if True, use learned convolutions for upsampling and
		downsampling.
	:param dims: determines if the signal is 1D, 2D, or 3D.
	:param num_classes: if specified (as an int), then this model will be
		class-conditional with `num_classes` classes.
	:param use_checkpoint: use gradient checkpointing to reduce memory usage.
	:param num_heads: the number of attention heads in each attention layer.
	:param num_heads_channels: if specified, ignore num_heads and instead use
							   a fixed channel width per attention head.
	:param num_heads_upsample: works with num_heads to set a different number
							   of heads for upsampling. Deprecated.
	:param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
	:param resblock_updown: use residual blocks for up/downsampling.
	:param use_new_attention_order: use a different attention pattern for potentially
									increased efficiency.
	"""

	def __init__(
		self,
		image_size,
		in_channels,
		model_channels,
		out_channels,
		num_res_blocks,
		attention_resolutions,
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

		self.image_size = image_size
		self.in_channels = in_channels
		self.model_channels = model_channels
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

		time_embed_dim = model_channels * 4
		self.time_embed = nn.Sequential(
			linear(model_channels, time_embed_dim),
			nn.SiLU(),
			linear(time_embed_dim, time_embed_dim),
		)

		if self.num_classes is not None:
			self.label_emb = nn.Embedding(num_classes, time_embed_dim)

		ch = input_ch = int(channel_mult[0] * model_channels)
		self.input_blocks = nn.ModuleList(
			[TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
		)
		self._feature_size = ch
		input_block_chans = [ch]
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
				input_block_chans.append(ch)
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
				input_block_chans.append(ch)
				ds *= 2
				self._feature_size += ch

		self.middle_block = TimestepEmbedSequential(
			ResBlock(
				ch,
				time_embed_dim,
				dropout,
				dims=dims,
				use_checkpoint=use_checkpoint,
				use_scale_shift_norm=use_scale_shift_norm,
			),
			AttentionBlock(
				ch,
				use_checkpoint=use_checkpoint,
				num_heads=num_heads,
				num_head_channels=num_head_channels,
				use_new_attention_order=use_new_attention_order,
			),
			ResBlock(
				ch,
				time_embed_dim,
				dropout,
				dims=dims,
				use_checkpoint=use_checkpoint,
				use_scale_shift_norm=use_scale_shift_norm,
			),
		)
		self._feature_size += ch
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
				self._feature_size += ch

		self.out = nn.Sequential(
			normalization(ch),
			nn.SiLU(),
			zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
		)

	def convert_to_fp16(self):
		"""
		Convert the torso of the model to float16.
		"""
		self.input_blocks.apply(convert_module_to_f16)
		self.middle_block.apply(convert_module_to_f16)
		self.output_blocks.apply(convert_module_to_f16)

	def convert_to_fp32(self):
		"""
		Convert the torso of the model to float32.
		"""
		self.input_blocks.apply(convert_module_to_f32)
		self.middle_block.apply(convert_module_to_f32)
		self.output_blocks.apply(convert_module_to_f32)

	def forward(self, x, timesteps, y=None):
		"""
		Apply the model to an input batch.

		:param x: an [N x C x ...] Tensor of inputs.
		:param timesteps: a 1-D batch of timesteps.
		:param y: an [N] Tensor of labels, if class-conditional.
		:return: an [N x C x ...] Tensor of outputs.
		"""
		assert (y is not None) == (
			self.num_classes is not None
		), "must specify y if and only if the model is class-conditional"

		hs = []
		emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

		if self.num_classes is not None:
			assert y.shape == (x.shape[0],)
			emb = emb + self.label_emb(y)
		h = x.type(self.dtype)
		for module in self.input_blocks:
			h = module(h, emb)
			hs.append(h)
		h = self.middle_block(h, emb)
		for module in self.output_blocks:
			h = th.cat([h, hs.pop()], dim=1)
			h = module(h, emb)
		h = h.type(x.dtype)
		
		return self.out(h)

class SuperResModel(UNetModel):
	"""
	A UNetModel that performs super-resolution.

	Expects an extra kwarg `low_res` to condition on a low-resolution image.
	"""

	def __init__(self, image_size, in_channels, *args, **kwargs):
		super().__init__(image_size, in_channels * 2, *args, **kwargs)

	def forward(self, x, timesteps, low_res=None, **kwargs):
		_, _, new_height, new_width = x.shape
		upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
		x = th.cat([x, upsampled], dim=1)
		return super().forward(x, timesteps, **kwargs)


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
		channel_mult=(1, 1, 2, 2, 4, 8),
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
		pool="adaptive",
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
		
		return hs


class DecoderUNetModel(nn.Module):
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
		pool="adaptive",
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
				self._feature_size += ch

		

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
		h = x.type(self.dtype)
		
		for module in self.input_blocks:
			h = module(h, emb)
			hs.append(h)
		
		return hs




class KernelUNet(nn.Module):
	"""
	The full UNet model with 2 inputs - kernel 64 X 64 and image 256 X 256
	aims to sample from distribution p(h|y)

	:param in_channels: channels in the input Tensor.
	:param model_channels: base channel count for the model.
	:param out_channels: channels in the output Tensor.
	:param num_res_blocks: number of residual blocks per downsample.
	:param attention_resolutions: a collection of downsample rates at which
		attention will take place. May be a set, list, or tuple.
		For example, if this contains 4, then at 4x downsampling, attention
		will be used.
	:param dropout: the dropout probability.
	:param channel_mult: channel multiplier for each level of the UNet.
	:param conv_resample: if True, use learned convolutions for upsampling and
		downsampling.
	:param dims: determines if the signal is 1D, 2D, or 3D.
	:param num_classes: if specified (as an int), then this model will be
		class-conditional with `num_classes` classes.
	:param use_checkpoint: use gradient checkpointing to reduce memory usage.
	:param num_heads: the number of attention heads in each attention layer.
	:param num_heads_channels: if specified, ignore num_heads and instead use
							   a fixed channel width per attention head.
	:param num_heads_upsample: works with num_heads to set a different number
							   of heads for upsampling. Deprecated.
	:param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
	:param resblock_updown: use residual blocks for up/downsampling.
	:param use_new_attention_order: use a different attention pattern for potentially
									increased efficiency.
	"""

	def __init__(
		self,
		kernel_size = 64,
		image_size = 256,
		in_channels = 3,
		model_channels = 64,
		out_channels = 512,
		num_res_blocks = 1,
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
		self.model_channels = model_channels
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


		common_kwargs = {
		'model_channels': model_channels,
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
		
		out_ch = int(model_channels*channel_mult[-1])
		self.encoder_kernel = EncoderUNetModel(image_size = kernel_size, in_channels = 1, 
			upscale_factor = int(image_size//kernel_size) , **common_kwargs)
		

		self.encoder_image = EncoderUNetModel(image_size = image_size, in_channels = in_channels, 
			upscale_factor = None, **common_kwargs)

		ch = 2*out_ch


	def convert_to_fp16(self):
		"""
		Convert the torso of the model to float16.
		"""
		self.encoder_kernel.apply(convert_module_to_f16)
		self.encoder_image.apply(convert_module_to_f16)
		# self.output_blocks.apply(convert_module_to_f16)

	def convert_to_fp32(self):
		"""
		Convert the torso of the model to float32.
		"""
		self.encoder_kernel.apply(convert_module_to_f32)
		self.encoder_image.apply(convert_module_to_f32)
		# self.output_blocks.apply(convert_module_to_f32)

	def forward(self, h, y, timesteps):
		"""
		Apply the model to an input batch.

		:param h: an [N x C x ...] Tensor of kernel inputs.
		:param y: an [N] Tensor of blurred images.
		:param timesteps: a 1-D batch of timesteps.
		:return: an [N x C x ...] Tensor of outputs.
		"""
		h_feat = self.encoder_kernel(h, timesteps)
		im_feat = self.encoder_image(y, timesteps)
		return h_feat, im_feat