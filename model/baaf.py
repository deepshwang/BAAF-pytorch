import torch
import torch.nn as nn
from utils.point_util import *
import time
import pdb


class MLP1d(nn.Module):
	def __init__(self, d_in, d_out, bias=True, bn=True, activation='relu'):
		super(MLP1d, self).__init__()
		self.linear = nn.Conv1d(d_in,
								d_out, 
								kernel_size=1,
								bias=bias)
		
		if bn:
			self.bn = nn.BatchNorm1d(d_out)
		else:
		 self.bn = None
		
		if activation == 'lrelu':
			self.act = nn.LeakyReLU(0.2)
		elif activation == 'relu':
			self.act = nn.ReLU()
		else:
			self.act=None

	def forward(self, x):
		"""
		Input: (B, N, d_in)
		Output: (B, N, d_out)
		"""
		x = x.transpose(1, 2).contiguous() # (B, d_in, N)
		x = self.linear(x) # (B, d_out, N)
		if self.bn is not None:
			x = self.bn(x)
		if self.act is not None:
			x = self.act(x)
		x = x.transpose(1, 2).contiguous() # (B, N, d_out)
		return x


class MLP2d(nn.Module):
	def __init__(self, d_in, d_out, bias=True, bn=True, activation='relu'):
		super(MLP2d, self).__init__()
		self.linear = nn.Conv2d(d_in,
								d_out, 
								kernel_size=1,
								bias=bias)
		
		if bn:
			self.bn = nn.BatchNorm2d(d_out)
		else:
		 self.bn = None
		
		if activation == 'lrelu':
			self.act = nn.LeakyReLU(0.2)
		elif activation == 'relu':
			self.act = nn.ReLU()
		else:
			self.act=None

	def forward(self, x):
		"""
		Input: (B, N, k, d_in)
		Output: (B, N, k, d_out)
		"""
		x = x.transpose(1, 3).contiguous() # (B, d_in, k, N)
		x = self.linear(x) # (B, d_out, k, N)
		if self.bn is not None:
			x = self.bn(x)
		if self.act is not None:
			x = self.act(x)
		x = x.transpose(1, 3).contiguous() # (B, N, k, d_out)
		return x


class MLP1dTrans(nn.Module):
	def __init__(self, d_in, d_out, bias=True, bn=True, activation='relu'):
		super(MLP1dTrans, self).__init__()
		self.conv_trans = nn.ConvTranspose1d(d_in, d_out, 1, bias=bias)
		
		if bn:
			self.bn = nn.BatchNorm1d(d_out)
		else:
		 self.bn = None
		
		if activation == 'lrelu':
			self.act = nn.LeakyReLU(0.2)
		elif activation == 'relu':
			self.act = nn.ReLU()
		else:
			self.act=None


	def forward(self, x):
		x = x.transpose(1, 2).contiguous() # (B, d_in, N)
		x = self.conv_trans(x) # (B, d_out, k, N)
		if self.bn is not None:
			x = self.bn(x)
		if self.act is not None:
			x = self.act(x)
		x = x.transpose(1, 2).contiguous() # (B, N, d_out)
		return x



class MLP2dTrans(nn.Module):
	def __init__(self, d_in, d_out, bias=True, bn=True, activation='relu'):
		super(MLP2dTrans, self).__init__()
		self.conv_trans = nn.ConvTranspose2d(d_in, d_out, 1, bias=bias)
		
		if bn:
			self.bn = nn.BatchNorm2d(d_out)
		else:
		 self.bn = None
		
		if activation == 'lrelu':
			self.act = nn.LeakyReLU(0.2)
		elif activation == 'relu':
			self.act = nn.ReLU()
		else:
			self.act=None


	def forward(self, x):
		x = x.transpose(1, 3).contiguous() # (B, d_in, k, N)
		x = self.conv_trans(x) # (B, d_out, k, N)
		if self.bn is not None:
			x = self.bn(x)
		if self.act is not None:
			x = self.act(x)
		x = x.transpose(1, 3).contiguous() # (B, N, k, d_out)
		return x



class BilateralAugmentation(nn.Module):
	def __init__(self, d_in, d_out, k=16):

		"""
		Bilateral Augmentation Block
		"""
		super(BilateralAugmentation, self).__init__()
		self.k = k
		self.d_in = d_in
		self.d_out = d_out
		self.mlp0 = MLP1d(d_in, d_out//2)
		self.mlp1 = MLP2d(d_out, 3)
		self.mlp2 = MLP2d(9, d_out // 2)
		self.mlp3 = MLP2d(9, d_out // 2)
		self.mlp4 = MLP2d(3*d_out//2, d_out // 2)


	def forward(self, p, f):
		"""
		Args:
			p [Points]: (B, N, 3)
			f [Features]: (B, N, d_in)
		
		Return:
			alc [Augmented Local Context / refer to 3.1 in paper]: (B, N, k, d_out)
		"""

		knn_idx = knn(p, p, self.k)
		# Encode feature input (Not mentioned in the paper, but implemented in the author's GIT)
		f = self.mlp0(f) # (B, N, d_out // 2)
		
		# Gather Euclidean k-Nearest-Neighbors 
		p_knn = index2kNNPoints(p, knn_idx) # (B, N, k, 3)
		f_knn = index2kNNPoints(f, knn_idx) # (B, N, k, d_out // 2)

		# Local Geometric Context
		lgc = self._embedContext(p, p_knn) # (B, N, k, 6)

		# Local Semantic Context
		lsc = self._embedContext(f, f_knn) # (B, N, k, d_out)

		# Augment Local Geometric Context
		p_knn_offset = self.mlp1(lsc)
		p_knn_tilde = p_knn_offset + p_knn
		lgc_aug = torch.cat((lgc, p_knn_tilde), dim=-1) # (B, N, k, 9)

		# Augmented Local Semantic Context (lsc_aug)
		f_knn_offset = self.mlp2(lgc_aug) # (B, N, k, d_out // 2)
		f_knn_tilde = f_knn_offset + f_knn
		lsc_aug = torch.cat((lsc, f_knn_tilde), dim=-1) # (B, N, k, 3/2*d_out)

		# Augmented Local Context (alc)
		p_knn_encode = self.mlp3(lgc_aug) # (B, N, k, d_out // 2)
		f_knn_encode = self.mlp4(lsc_aug) # (B, N, k, d_out // 2)
		alc = torch.cat((p_knn_encode, f_knn_encode), dim=-1) # (B, N, k, d_out)
		
		return alc, p_knn_tilde

	def _embedContext(self, x, x_knn): 
		x_expanded = torch.unsqueeze(x, dim=2).expand(-1, -1, self.k, -1)
		x_rel = x_knn - x_expanded
		embed = torch.cat((x_expanded, x_rel), dim=-1)
		return embed


class MixedLocalAggregation(nn.Module):
	def __init__(self, d):
		"""
		Mixed Local Aggregation Block
		"""
		super(MixedLocalAggregation, self).__init__()
		self.mlp0 = MLP2d(d, d, bn=False, activation=False)
		self.mlp1 = MLP2d(2*d, d)
		self.mlp2 = MLP2d(d, 2*d, activation='lrelu')

	def forward(self, alc):
		"""
		Args:
			alc [Aggregated Local Context]: (B, N, k, d)
		
		Return:
			mla [Mixed Local Aggregation]: (B, N, k, 2*d)

		"""
		k_weights = self.mlp0(alc) # (B, N, k, d)
		k_weights = nn.functional.softmax(k_weights, dim=2) # (B, N, k, d)
		alc_weighted_sum = torch.sum(alc * k_weights, dim=2, keepdim=True) # (B, N, 1, d)
		alc_max = torch.max(alc, axis=2, keepdims=True)[0] # (B, N, 1, d)
		mla = torch.cat((alc_weighted_sum, alc_max), dim=-1) # (B, N, 1, 2*d)
		mla = self.mlp1(mla) # (B, N, 1, d)
		mla = self.mlp2(mla) # (B, N, 1, 2*d)
		mla = torch.squeeze(mla)
		return mla


class BilateralContextBlock(nn.Module):
	def __init__(self, d_in, d_out, k):
		"""
		Bilateral Context Block
		"""
		super(BilateralContextBlock, self).__init__()
		self.BA = BilateralAugmentation(d_in, d_out, k)
		self.MLA = MixedLocalAggregation(d_out)

	def forward(self, p, f):
		"""
		Args:
			p [point]: (B, N, 3)
			f [feature]: (B, N, d_in)

		Return:
			f [output feature]: (B, N, 2 * d_in)
		"""
		f, p_knn_tilde = self.BA(p, f)
		f = self.MLA(f)
		return f, p_knn_tilde


class BAAFNet(nn.Module):
	def __init__(self, n_points=4096, ds_ratio=4, k=16, num_classes=13, dims=None):
		"""
		Bilateral Augmentation and Adaptive Fusion Network (BAAF-Net)
		"""
		super(BAAFNet, self).__init__()
		self.n_points = n_points
		self.ds_ratio = ds_ratio
		self.k = k
		self.dims = dims[1:]
		self.num_layers = len(self.dims) - 1
		# Feature Embed
		self.mlp0 = MLP1d(dims[0], dims[1] * 2, bn=True, activation='lrelu')
		# Encoders
		self.EncoderBCBModules = nn.ModuleList([BilateralContextBlock(self.dims[i] * 2, self.dims[i+1], k=k) for i in range(len(self.dims) - 1)])
		# Decoders
		self.DecoderMLPModules = nn.ModuleList([MLP1d(2* self.dims[-1-i], 2 * self.dims[-1-i]) for i in range(self.num_layers)])
		self.DecoderReconModules = self._getDecoderReconModules()
		self.DecoderWeightModules = nn.ModuleList([MLP1d(2*self.dims[0], 1, bn=False, activation=None) for i in range(self.num_layers)])
		# Classifier
		self.classifier = nn.Sequential(MLP1d(2 * self.dims[0], 64),
										MLP1d(64, 32),
										nn.Dropout(p=0.5),
										MLP1d(32, num_classes))


	def forward(self, p, f):
		start = time.time()
		n_points = self.n_points
		f_encoder_list = []
		p_list = []
		p_knn_tilde_list = []
		p_ds_list = []
		ds_idx_list = []
		us_idx_list = []
		
		## Initial Feature Embedding
		f = self.mlp0(f)


		#############################################
		### Encoding <<Bilateral Context Module>> ###
		#############################################
		for i in range(self.num_layers):
			f, p_knn_tilde = self.EncoderBCBModules[i](p, f)
			p_knn_tilde_list.append(p_knn_tilde)
			if i == 0:
				f_encoder_list.append(f)


			# Downsample p and f using FPS (Farthest Point Sampling)
			n_points = n_points // self.ds_ratio
			ds_idx = farthest_point_sample(p, n_points)
			p_ds = index2Points(p, ds_idx)
			f = index2Points(f, ds_idx)
			f_encoder_list.append(f)			
			
			# Save upsampling index for interpolation
			us_idx = knn(db=p_ds, q=p, k=1)
			us_idx_list.append(us_idx)

			# Save original and downsampled points
			p_list.append(p)
			p_ds_list.append(p_ds)

			p = p_ds


		#############################################	
		#### Decoding <<Adaptive Fusion Module>> ####
		#############################################
		
		f_decoder_list = []
		f_decoder_weights_list = []
		for n in range(self.num_layers):
			f = f_encoder_list[-1-n]
			f = self.DecoderMLPModules[n](f)

			for j in range(self.num_layers-n):
				f_interp_i = nearest_interpolation(f, us_idx_list[-j-n-1])
				f_cat = torch.cat((f_encoder_list[-j-n-2], f_interp_i), axis=-1)
				f = self.DecoderReconModules[n][j](f_cat)

			f_decoder_list.append(f)
			curr_weight = self.DecoderWeightModules[n](f)
			f_decoder_weights_list.append(curr_weight)

		f_weights = torch.cat(f_decoder_weights_list, axis=-1)
		f_weights = nn.functional.softmax(f_weights, dim=-1)

		f_weights_list = torch.split(f_weights, 1, dim=2)
		weighted_f_list = []
		for f, w in zip(f_decoder_list, f_weights_list):
			weighted_f_list.append(f*w)
		f = torch.stack(weighted_f_list, dim=0).sum(dim=0)


		## Final Semantic Classification 
		out = self.classifier(f)

		return out, p_knn_tilde_list, p_list


	def _getDecoderReconModules(self):
		total_module_list = []
		
		for n in range(self.num_layers):
			layer_module_list = []
			for j in range(self.num_layers - n):
				d_in = 2 * (self.dims[-1-n-j] + self.dims[-2-n-j])
				d_out = 2 * self.dims[-2-n-j]
				if j+1 == self.num_layers-n:
					d_in = 4 * self.dims[-1-n-j]
				layer_module_list.append(MLP1dTrans(d_in, d_out))
			layer = nn.ModuleList(layer_module_list)
			total_module_list.append(layer)

		total_module = nn.ModuleList(total_module_list)
		return total_module
