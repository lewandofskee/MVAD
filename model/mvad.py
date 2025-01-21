import torch
import torch.nn as nn
try:
	from torch.hub import load_state_dict_from_url
except ImportError:
	from torch.utils.model_zoo import load_url as load_state_dict_from_url
from timm.models.resnet import Bottleneck

from model import get_model
from model import MODEL
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
import copy
import math
from timm.models.layers import DropPath

def build_position_embedding(pos_embed_type, feature_size, hidden_dim):
	if pos_embed_type in ("v2", "sine"):
		# TODO find a better way of exposing other arguments
		pos_embed = PositionEmbeddingSine(feature_size, hidden_dim // 2, normalize=True)
	elif pos_embed_type in ("v3", "learned"):
		pos_embed = PositionEmbeddingLearned(feature_size, hidden_dim // 2)
	else:
		raise ValueError(f"not supported {pos_embed_type}")
	return pos_embed

class PositionEmbeddingSine(nn.Module):
	"""
	This is a more standard version of the position embedding, very similar to the one
	used by the Attention is all you need paper, generalized to work on images.
	"""

	def __init__(
		self,
		feature_size,
		num_pos_feats=128,
		temperature=10000,
		normalize=False,
		scale=None,
	):
		super().__init__()
		self.feature_size = feature_size
		self.num_pos_feats = num_pos_feats
		self.temperature = temperature
		self.normalize = normalize
		if scale is not None and normalize is False:
			raise ValueError("normalize should be True if scale is passed")
		if scale is None:
			scale = 2 * math.pi
		self.scale = scale

	def forward(self, tensor):
		not_mask = torch.ones((self.feature_size[0], self.feature_size[1]))  # H x W
		y_embed = not_mask.cumsum(0, dtype=torch.float32)
		x_embed = not_mask.cumsum(1, dtype=torch.float32)
		if self.normalize:
			eps = 1e-6
			y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
			x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

		dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
		dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

		pos_x = x_embed[:, :, None] / dim_t
		pos_y = y_embed[:, :, None] / dim_t
		pos_x = torch.stack(
			(pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
		).flatten(2)
		pos_y = torch.stack(
			(pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
		).flatten(2)
		pos = torch.cat((pos_y, pos_x), dim=2).flatten(0, 1)  # (H X W) X C
		return pos.to(tensor.device)


# ========== Decoder ==========
def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride = 1) -> nn.Conv2d:
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def deconv2x2(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
	return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride, groups=groups, bias=False, dilation=dilation)


class DeBasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride = 1, upsample = None, groups = 1, base_width = 64,
		dilation = 1, norm_layer = None):
		super(DeBasicBlock, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		if groups != 1 or base_width != 64:
			raise ValueError('BasicBlock only supports groups=1 and base_width=64')
		if dilation > 1:
			raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
		# Both self.conv1 and self.downsample layers downsample the input when stride != 1
		if stride == 2:
			self.conv1 = deconv2x2(inplanes, planes, stride)
		else:
			self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = norm_layer(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = norm_layer(planes)
		self.upsample = upsample
		self.stride = stride

	def forward(self, x):
		identity = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		if self.upsample is not None:
			identity = self.upsample(x)
		out += identity
		out = self.relu(out)
		return out


class DeBottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride = 1, upsample = None, groups = 1, base_width = 64,
		dilation = 1, norm_layer = None):
		super(DeBottleneck, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		width = int(planes * (base_width / 64.)) * groups
		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv1x1(inplanes, width)
		self.bn1 = norm_layer(width)
		if stride == 2:
			self.conv2 = deconv2x2(width, width, stride, groups, dilation)
		else:
			self.conv2 = conv3x3(width, width, stride, groups, dilation)
		self.bn2 = norm_layer(width)
		self.conv3 = conv1x1(width, planes * self.expansion)
		self.bn3 = norm_layer(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.upsample = upsample
		self.stride = stride

	def forward(self, x):
		identity = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)
		out = self.conv3(out)
		out = self.bn3(out)
		if self.upsample is not None:
			identity = self.upsample(x)
		out += identity
		out = self.relu(out)
		return out


class ResNet(nn.Module):
	def __init__(self, block, layers, num_classes = 1000,
				 zero_init_residual = False, groups = 1, width_per_group = 64, replace_stride_with_dilation = None,
				 norm_layer = None ):
		super(ResNet, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		self._norm_layer = norm_layer

		self.inplanes = 512 * block.expansion
		self.dilation = 1
		if replace_stride_with_dilation is None:
			replace_stride_with_dilation = [False, False, False]
		if len(replace_stride_with_dilation) != 3:
			raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(replace_stride_with_dilation))
		self.groups = groups
		self.base_width = width_per_group
		self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
		self.layer3 = self._make_layer(block, 64, layers[2], stride=2, dilate=replace_stride_with_dilation[1])

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, DeBottleneck):
					nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
				elif isinstance(m, DeBasicBlock):
					nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

	def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
		norm_layer = self._norm_layer
		upsample = None
		previous_dilation = self.dilation
		if dilate:
			self.dilation *= stride
			stride = 1
		if stride != 1 or self.inplanes != planes * block.expansion:
			upsample = nn.Sequential(deconv2x2(self.inplanes, planes * block.expansion, stride),
									 norm_layer(planes * block.expansion),)
		layers = []
		layers.append(block(self.inplanes, planes, stride, upsample, self.groups, self.base_width, previous_dilation, norm_layer))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
		return nn.Sequential(*layers)

	def _forward_impl(self, x):
		feature_a = self.layer1(x)  # 512*8*8->256*16*16
		feature_b = self.layer2(feature_a)  # 256*16*16->128*32*32
		feature_c = self.layer3(feature_b)  # 128*32*32->64*64*64
		return [feature_c, feature_b, feature_a]

	def forward(self, x):
		return self._forward_impl(x)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
	model = ResNet(block, layers, **kwargs)
	return model

@MODEL.register_module
def de_resnet34(pretrained = False, progress = True, **kwargs):
	return _resnet('resnet34', DeBasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)

class TopK(nn.Module):
	def __init__(self, qk_dim, topk=4, qk_scale=None,):
		super().__init__()
		self.topk = topk
		self.qk_dim = qk_dim
		self.scale = qk_scale or qk_dim ** -0.5
		self.emb =  nn.Identity()
		self.act = nn.Softmax(dim=-1)

	def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor]:
		query, key = query.detach(), key.detach()
		query_hat, key_hat = self.emb(query), self.emb(key)  # (n, p^2, c) (n, 4, p^2, c)
		key_hat = rearrange(key_hat, 'n v p c -> n (v p) c')
		attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)  # (n, p^2, 4*p^2)
		topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)  # (n, p^2, k), (n, p^2, k)
		r_weight = self.act(topk_attn_logit)  # (n, p^2, k)

		return r_weight, topk_index

class KVG(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, r_idx:Tensor, r_weight:Tensor, kv:Tensor):
		n, v, p2, w2, c_kv = kv.size() #(bs,4,aa,hw/aa,c)
		topk = r_idx.size(-1) #(bs,4,aa,topk)
		topk_kv = torch.gather(kv.view(n, 1, v * p2, w2, c_kv).expand(-1, p2, -1, -1, -1), # (n, p^2, 4 * p^2, w^2, c_kv)
								dim=2,
								index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv) # (n, p^2, topk, w^2, c_kv)
							   ) # (n, 4, p^2, topk, w^2, c_kv)
		return topk_kv

class MVAS(nn.Module):
	def __init__(self, d_model, n_win, num_heads, topk, auto_pad=True,):
		super().__init__()
		self.d_model = d_model
		self.n_win = n_win
		self.num_heads = num_heads
		self.topk = topk
		self.auto_pad = auto_pad
		self.scale = self.d_model ** -0.5
		self.ftopk = TopK(qk_dim=self.d_model, qk_scale=self.scale, topk=self.topk,)
		self.kv_gather = KVG()
		self.attn_act = nn.Softmax(dim=-1)

	def forward(self, cv_feature, mv_feature):
		if self.auto_pad:
			N, H_in, W_in, C = cv_feature.size()
			pad_l = pad_t = 0
			pad_r = (self.n_win - W_in % self.n_win) % self.n_win
			pad_b = (self.n_win - H_in % self.n_win) % self.n_win
			cv_feature_p = F.pad(cv_feature, (0, 0,  # dim=-1
						  pad_l, pad_r,  # dim=-2
						  pad_t, pad_b))  # dim=-3
			mv_feature = F.pad(mv_feature, (0, 0,  # dim=-1
											pad_l, pad_r,  # dim=-2
											pad_t, pad_b))  # dim=-3
			_, H, W, _ = cv_feature_p.size()  # padded size
		else:
			N, H, W, C = cv_feature.size()
			cv_feature_p = cv_feature
			assert H % self.n_win == 0 and W % self.n_win == 0
		cv_patch = rearrange(cv_feature_p, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win) #(bs,h,w,c) -> (bs,aa,h/a,w/a,c)
		mv_patch = rearrange(mv_feature, "n v (j h) (i w) c -> n v (j i) h w c", j=self.n_win, i=self.n_win) #(bs,4,h,w,c) -> (bs,4,aa,h/a,w/a,c)
		q_pix = rearrange(cv_patch, 'n p2 h w c -> n p2 (h w) c') #(bs,aa,hw/aa,c)
		kv_pix = rearrange(mv_patch, 'n v p2 h w c -> n v p2 (h w) c') #(bs,4,aa,hw/aa,c)

		q_win, k_win = cv_patch.mean([2, 3]), mv_patch.mean([3, 4]) #q:(bs,aa,c), k:(bs,4,aa,c)
		r_weight, r_idx = self.ftopk(q_win, k_win) #(bs,4,aa,topk)
		kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix) #(bs,4,aa,topk,hw/aa,c)
		k_pix_sel = v_pix_sel = kv_pix_sel #(bs,4,aa,topk,hw/aa,c)
		k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)',
							  m=self.num_heads)  # flatten to BMLC, (n*aa, 4, m, c//m, topk*hw/aa)
		v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c',
							  m=self.num_heads)  # flatten to BMLC, (n*aa, 4, m, topk*hw/aa, c//m)
		q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c',
						  m=self.num_heads)  # to BMLC tensor (n*p^2, 1, m, w^2, c//m)
		attn_weight = (q_pix * self.scale) @ k_pix_sel  # (n*p^2, 1, m, w^2, c) @ (n*p^2, 4, m, c, topk*h_kv*w_kv) -> (n*p^2, 4, m, w^2, topk*h_kv*w_kv)
		attn_weight = self.attn_act(attn_weight) #(n*p^2, 4, m, w^2, topk*h_kv*w_kv)
		out = attn_weight @ v_pix_sel  # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, 4, m, w^2, c)
		out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
						h=H // self.n_win, w=W // self.n_win) #(bs,4,h,w,c)
		if self.auto_pad and (pad_r > 0 or pad_b > 0):
			out = out[:, :H_in, :W_in, :].contiguous()
		return out


class MID_LAYER(nn.Module):
	def __init__(self, block, layers, width_per_group = 64, windows=[8,8,8], topks=[16,32,64], norm_layer = None,):
		super(MID_LAYER, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		self._norm_layer = norm_layer
		self.base_width = width_per_group
		base = 16
		self.inplanes = base * 4 * block.expansion
		self.dilation = 1
		self.bn_layer = self._make_layer(block, base * 8, layers, stride=2)

		self.conv1 = conv3x3(base * block.expansion, base * 2 * block.expansion, 2)
		self.bn1 = norm_layer(base * 2 * block.expansion)
		self.conv2 = conv3x3(base * 2 * block.expansion, base * 4 * block.expansion, 2)
		self.bn2 = norm_layer(base * 4 * block.expansion)
		self.conv21 = nn.Conv2d(base * 2 * block.expansion, base * 2 * block.expansion, 1)
		self.bn21 = norm_layer(base * 2 * block.expansion)
		self.conv31 = nn.Conv2d(base * 4 * block.expansion, base * 4 * block.expansion, 1)
		self.bn31 = norm_layer(base * 4 * block.expansion)
		self.convf = nn.Conv2d(base * 4 * block.expansion, base * 4 * block.expansion, 1)
		self.bnf = norm_layer(base * 4 * block.expansion)
		self.relu = nn.ReLU(inplace=True)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
		self.drop_rate = 0.1
		self.drop_path = DropPath(self.drop_rate) if self.drop_rate > 0. else nn.Identity()

		### layer 0
		self.mvas0 = MVAS(d_model=base * block.expansion, n_win=windows[0], num_heads=8, topk=topks[0])
		self.norm01 = nn.LayerNorm(base * block.expansion, eps=1e-6)
		self.mlp0 = nn.Sequential(nn.Linear(base * block.expansion, base * 4 * block.expansion),
								  nn.ReLU(),
								  nn.Dropout(self.drop_rate),
								  nn.Linear(base * 4 * block.expansion, base * block.expansion),
								  nn.Dropout(self.drop_rate),)
		self.norm02 = nn.LayerNorm(base * block.expansion, eps=1e-6)

		### layer 1
		self.mvas1 = MVAS(d_model=base * 2 * block.expansion, n_win=windows[1], num_heads=8, topk=topks[1])
		self.norm11 = nn.LayerNorm(base * 2 * block.expansion, eps=1e-6)
		self.mlp1 = nn.Sequential(nn.Linear(base * 2 * block.expansion, base * 8 * block.expansion),
								  nn.ReLU(),
								  nn.Dropout(self.drop_rate),
								  nn.Linear(base * 8 * block.expansion, base * 2 * block.expansion),
								  nn.Dropout(self.drop_rate),)
		self.norm12 = nn.LayerNorm(base * 2 * block.expansion, eps=1e-6)
		self.mvas11 = MVAS(d_model=base * 2 * block.expansion, n_win=windows[1], num_heads=8, topk=topks[1])
		self.norm111 = nn.LayerNorm(base * 2 * block.expansion, eps=1e-6)
		self.mlp11 = nn.Sequential(nn.Linear(base * 2 * block.expansion, base * 8 * block.expansion),
								  nn.ReLU(),
								  nn.Dropout(self.drop_rate),
								  nn.Linear(base * 8 * block.expansion, base * 2 * block.expansion),
								  nn.Dropout(self.drop_rate), )
		self.norm121 = nn.LayerNorm(base * 2 * block.expansion, eps=1e-6)

		### layer 2
		self.mvas2 = MVAS(d_model=base * 4 * block.expansion, n_win=windows[2], num_heads=8, topk=topks[2])
		self.norm21 = nn.LayerNorm(base * 4 * block.expansion, eps=1e-6)
		self.mlp2 = nn.Sequential(nn.Linear(base * 4 * block.expansion, base * 16 * block.expansion),
                                 nn.ReLU(),
								  nn.Dropout(self.drop_rate),
                                 nn.Linear(base * 16 * block.expansion, base * 4 * block.expansion),
								  nn.Dropout(self.drop_rate),)
		self.norm22 = nn.LayerNorm(base * 4 * block.expansion, eps=1e-6)
		self.mvas21 = MVAS(d_model=base * 4 * block.expansion, n_win=windows[2], num_heads=8, topk=topks[2])
		self.norm211 = nn.LayerNorm(base * 4 * block.expansion, eps=1e-6)
		self.mlp21 = nn.Sequential(nn.Linear(base * 4 * block.expansion, base * 16 * block.expansion),
								  nn.ReLU(),
								  nn.Dropout(self.drop_rate),
								  nn.Linear(base * 16 * block.expansion, base * 4 * block.expansion),
								  nn.Dropout(self.drop_rate), )
		self.norm221 = nn.LayerNorm(base * 4 * block.expansion, eps=1e-6)
		self.mvas22 = MVAS(d_model=base * 4 * block.expansion, n_win=windows[2], num_heads=8, topk=topks[2])
		self.norm212 = nn.LayerNorm(base * 4 * block.expansion, eps=1e-6)
		self.mlp22 = nn.Sequential(nn.Linear(base * 4 * block.expansion, base * 16 * block.expansion),
								  nn.ReLU(),
								  nn.Dropout(self.drop_rate),
								  nn.Linear(base * 16 * block.expansion, base * 4 * block.expansion),
								  nn.Dropout(self.drop_rate), )
		self.norm222 = nn.LayerNorm(base * 4 * block.expansion, eps=1e-6)
		self.mvas23 = MVAS(d_model=base * 4 * block.expansion, n_win=windows[2], num_heads=8, topk=topks[2])
		self.norm213 = nn.LayerNorm(base * 4 * block.expansion, eps=1e-6)
		self.mlp23 = nn.Sequential(nn.Linear(base * 4 * block.expansion, base * 16 * block.expansion),
								  nn.ReLU(),
								  nn.Dropout(self.drop_rate),
								  nn.Linear(base * 16 * block.expansion, base * 4 * block.expansion),
								  nn.Dropout(self.drop_rate), )
		self.norm223 = nn.LayerNorm(base * 4 * block.expansion, eps=1e-6)
		self.pos_embed0 = build_position_embedding('sine', [64, 64], base * 4)
		self.pos_embed1 = build_position_embedding('sine', [32, 32], base * 8)
		self.pos_embed2 = build_position_embedding('sine', [16, 16], base * 16)

	def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
		norm_layer = self._norm_layer
		downsample = None
		previous_dilation = self.dilation
		if dilate:
			self.dilation *= stride
			stride = 1
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
									   norm_layer(planes * block.expansion), )
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
		return nn.Sequential(*layers)

	def with_pos_embed(self, tensor, pos: Optional[Tensor]):
		return tensor if pos is None else tensor + pos

	def forward(self, input):
		### layer 0
		mid_feature0 = rearrange(input[0], "(b f) c h w-> b f h w c", f=5).contiguous() #(bs/5, 5, 64, 64, 64)
		sv_features = []
		for i in range(mid_feature0.size(1)):
			cv_feature = mid_feature0[:, i, :]
			pos_embed = self.pos_embed0(cv_feature)
			pos_embed = rearrange(pos_embed, "(h w) c -> h w c", h=64, w=64).contiguous()
			pos_embed = torch.cat(
				[pos_embed.unsqueeze(0)] * cv_feature.size(0), dim=0
			)
			cv_feature_wp = cv_feature + pos_embed
			mv_feature = torch.cat([mid_feature0[:, :i], mid_feature0[:, i + 1:]], dim=1)
			x = self.norm01(cv_feature + self.drop_path(self.mvas0(cv_feature_wp, mv_feature)))  # (N, H, W, C)
			fused_cv_feat = self.norm02(x + self.mlp0(x))  # (N, H, W, C)
			sv_features.append(fused_cv_feat.unsqueeze(dim=1))
		enhance_sv_features = torch.cat(sv_features, dim=1)
		enhance_sv_features = rearrange(enhance_sv_features, "b f h w c -> (b f) c h w").contiguous()
		fpn0 = self.relu(self.bn1(self.conv1(enhance_sv_features))) #(bs, 128, 32, 32)

		### layer 1
		mid_feature1 = rearrange(input[1], "(b f) c h w-> b f h w c", f=5).contiguous() #(bs,128, 32,32)
		sv_features = []
		for i in range(mid_feature1.size(1)):
			cv_feature = mid_feature1[:, i, :]
			pos_embed = self.pos_embed1(cv_feature)
			pos_embed = rearrange(pos_embed, "(h w) c -> h w c", h=32, w=32).contiguous()
			pos_embed = torch.cat(
				[pos_embed.unsqueeze(0)] * cv_feature.size(0), dim=0
			)
			cv_feature_wp = cv_feature + pos_embed
			mv_feature = torch.cat([mid_feature1[:, :i], mid_feature1[:, i + 1:]], dim=1)
			x = self.norm11(cv_feature + self.drop_path(self.mvas1(cv_feature_wp, mv_feature)))  # (N, H, W, C)
			x = self.norm12(x + self.mlp1(x))  # (N, H, W, C)
			x = self.norm111(x + self.drop_path(self.mvas11(x+pos_embed, mv_feature)))  # (N, H, W, C)
			fused_cv_feat = self.norm121(x + self.mlp11(x))  # (N, H, W, C)
			sv_features.append(fused_cv_feat.unsqueeze(dim=1))
		enhance_sv_features = torch.cat(sv_features, dim=1)
		enhance_sv_features = rearrange(enhance_sv_features, "b f h w c -> (b f) c h w").contiguous()
		fpn1 = self.relu(self.bn21(self.conv21(enhance_sv_features))) + fpn0 #(bs,256, 16, 16)

		### layer 2
		mid_feature2 = rearrange(input[2], "(b f) c h w-> b f h w c", f=5).contiguous() #(bs,256,16,16)
		sv_features = []
		for i in range(mid_feature2.size(1)):
			cv_feature = mid_feature2[:, i, :]
			pos_embed = self.pos_embed2(cv_feature)
			pos_embed = rearrange(pos_embed, "(h w) c -> h w c", h=16, w=16).contiguous()
			pos_embed = torch.cat(
				[pos_embed.unsqueeze(0)] * cv_feature.size(0), dim=0
			)
			cv_feature_wp = cv_feature + pos_embed
			mv_feature = torch.cat([mid_feature2[:, :i], mid_feature2[:, i + 1:]], dim=1)
			x = self.norm21(cv_feature + self.drop_path(self.mvas2(cv_feature_wp, mv_feature)))  # (N, H, W, C)
			x = self.norm22(x + self.mlp2(x))  # (N, H, W, C)
			x = self.norm211(x + self.drop_path(self.mvas21(x+pos_embed, mv_feature)))  # (N, H, W, C)
			x = self.norm221(x + self.mlp21(x))  # (N, H, W, C)
			x = self.norm212(x + self.drop_path(self.mvas22(x+pos_embed, mv_feature)))  # (N, H, W, C)
			x = self.norm222(x + self.mlp22(x))  # (N, H, W, C)
			x = self.norm213(x + self.drop_path(self.mvas23(x+pos_embed, mv_feature)))  # (N, H, W, C)
			fused_cv_feat = self.norm223(x + self.mlp23(x))  # (N, H, W, C)
			sv_features.append(fused_cv_feat.unsqueeze(dim=1))
		enhance_sv_features = torch.cat(sv_features, dim=1)
		enhance_sv_features = rearrange(enhance_sv_features, "b f h w c -> (b f) c h w").contiguous()

		sv_features = self.relu(self.bn2(self.conv2(fpn1))) + self.relu(self.bn31(self.conv31(enhance_sv_features)))
		enhance_sv_features = self.relu(self.bnf(self.convf(sv_features)))
		output = self.bn_layer(enhance_sv_features)
		return output.contiguous()


class MVAD(nn.Module):
	def __init__(self, model_t, model_s,windows=[8,8,8], topks=[16,32,64],):
		super(MVAD, self).__init__()
		self.net_t = get_model(model_t)
		self.mff_oce = MID_LAYER(Bottleneck, 3, windows=windows, topks=topks,)
		self.net_s = get_model(model_s)

		self.frozen_layers = ['net_t']

	def freeze_layer(self, module):
		module.eval()
		for param in module.parameters():
			param.requires_grad = False

	def train(self, mode=True):
		self.training = mode
		for mname, module in self.named_children():
			if mname in self.frozen_layers:
				self.freeze_layer(module)
			else:
				module.train(mode)
		return self

	def forward(self, imgs):
		feats_t = self.net_t(imgs)
		feats_t = [f.detach() for f in feats_t]
		feats_s = self.net_s(self.mff_oce(feats_t))
		return feats_t, feats_s

@MODEL.register_module
def mvad(pretrained=False, **kwargs):
	model = MVAD(windows=[14,9,6], topks=[4,4,4],**kwargs)
	return model

if __name__ == '__main__':
	from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
	from util.util import get_timepc, get_net_params
	from argparse import Namespace
	bs = 10
	img_size = 256
	gpu_id = 0
	speed = True
	model_name = 'mvad'
	with torch.no_grad():
		x = torch.randn(bs, 3, img_size, img_size)
		model_t = Namespace()
		model_t.name = 'timm_resnet34'
		model_t.kwargs = dict(pretrained=False, checkpoint_path='model/pretrain/resnet34-43635321.pth',
								   strict=False, features_only=True, out_indices=[1, 2, 3])
		model_s = Namespace()
		model_s.name = 'de_resnet34'
		model_s.kwargs = dict(pretrained=False, checkpoint_path='', strict=False)
		model = Namespace()
		model.name = 'mvad'
		model.kwargs = dict(pretrained=False, checkpoint_path='', strict=True, model_t=model_t,
								 model_s=model_s)
		net = mvad(pretrained=False, model_t=model_t, model_s=model_s).cuda()
		net.eval()
		pre_cnt, cnt = 2, 5
		if gpu_id > -1:
			torch.cuda.set_device(gpu_id)
			x = x.cuda()
			net.cuda()
			pre_cnt, cnt = 5, 20
		flops = FlopCountAnalysis(net, x).total() / bs / 1e9
		params = parameter_count(net)[''] / 1e6
		flops_detail = FlopCountAnalysis(net,
										 torch.randn(5, 3, img_size, img_size).cuda() if gpu_id > -1 else torch.randn(5, 3, img_size,
																											img_size))
		print(flop_count_table(flops_detail, max_depth=3))
		if speed:
			# with torch.no_grad():
			for _ in range(pre_cnt):
				y = net(x)
			t_s = get_timepc()
			for _ in range(cnt):
				y = net(x)
			t_e = get_timepc()
			flops = f'{flops:>6.3f}'
			params = f'{params:>6.3f}'
			speed = f'{bs * cnt / (t_e - t_s):>7.3f}'
			ret_str = f'{model_name:>50}\t[GFLOPs: {flops}G]\t[Params: {params}M]\t[Speed: {speed}]\n'
			print(ret_str)
