from __future__ import annotations

import sys

import torch
from torch import nn
from torch.nn import functional as F

if len(sys.argv) < 2:
	sys.argv.append("cpu")

import EvoMSA2XYZ


def _valid_mask(coords: torch.Tensor) -> torch.Tensor:
	return torch.isfinite(coords).all(dim=-1)


class PairBiasModule(nn.Module):
	def __init__(self, residue_dim: int, pair_dim: int) -> None:
		super().__init__()
		hidden_dim = max(pair_dim, residue_dim)
		self.norm = nn.LayerNorm(residue_dim * 4)
		self.net = nn.Sequential(
			nn.Linear(residue_dim * 4, hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, pair_dim),
		)
		nn.init.zeros_(self.net[-1].weight)
		nn.init.zeros_(self.net[-1].bias)

	def forward(self, residue_embed: torch.Tensor) -> torch.Tensor:
		if residue_embed.ndim != 2:
			raise ValueError("Expected residue embeddings with shape [L, D].")
		left = residue_embed[:, None, :].expand(-1, residue_embed.shape[0], -1)
		right = residue_embed[None, :, :].expand(residue_embed.shape[0], -1, -1)
		pair_features = torch.cat(
			[left, right, left * right, torch.abs(left - right)], dim=-1
		)
		return self.net(self.norm(pair_features))


class ModifiedMSA2xyzIteration(EvoMSA2XYZ.MSA2xyzIteration):
	def __init__(
		self,
		seq_dim,
		msa_dim,
		N_ensemble,
		m_dim=64,
		s_dim=128,
		z_dim=64,
		docheck=True,
	):
		super().__init__(seq_dim, msa_dim, N_ensemble, m_dim=m_dim, s_dim=s_dim, z_dim=z_dim, docheck=docheck)
		self.pair_bias = PairBiasModule(m_dim, z_dim)

	def pred(self, msa_, idx, ss_, m1_pre, z_pre, pre_x, cycle_index, alphas, previous_dis, previous_hb):
		m1_all, z_all, s_all = 0, 0, 0
		N, L, _ = msa_.shape
		for i in range(self.N_ensemble):
			msa_mask = torch.zeros(N, L, device=msa_.device)
			msa_true = msa_ + 0
			seq = msa_true[0] * 1.0
			msa = torch.cat([msa_true * (1 - msa_mask[:, :, None]), msa_mask[:, :, None]], dim=-1)
			m, z = self.premsa(seq, msa, idx, alphas)
			if ss_ is None:
				ss = 0
			else:
				ss = torch.mean(self.pre_z(ss_), dim=0)
			z = z + ss
			pair_bias = self.pair_bias(m[0])
			m1_, z_ = self.re_emb(
				m1_pre, z_pre, pre_x, previous_dis, previous_hb, cycle_index == 0
			)
			z = z + z_ + pair_bias
			m = torch.cat([(m[0] + m1_)[None, ...], m[1:]], dim=0)
			m, z = self.evmodel(m, z)
			s = self.slinear(m[0])
			m1_all = m1_all + m[0]
			z_all = z_all + z
			s_all = s_all + s
		return m1_all / self.N_ensemble, z_all / self.N_ensemble, s_all / self.N_ensemble


class MSA2XYZ(EvoMSA2XYZ.MSA2XYZ):
	def __init__(
		self,
		seq_dim,
		msa_dim,
		N_ensemble,
		N_cycle,
		m_dim=64,
		s_dim=128,
		z_dim=64,
		docheck=True,
	):
		super().__init__(seq_dim, msa_dim, N_ensemble, N_cycle, m_dim=m_dim, s_dim=s_dim, z_dim=z_dim, docheck=docheck)
		self.msaxyzone = ModifiedMSA2xyzIteration(
			seq_dim,
			msa_dim,
			N_ensemble,
			m_dim=m_dim,
			s_dim=s_dim,
			z_dim=z_dim,
			docheck=docheck,
		)
		self.geometry_distance_threshold = 9.0
		self.geometry_loss_weight = 0.1

	def training_loss(
		self,
		pred_atom: torch.Tensor,
		ref_coords,
		coord_scaler=None,
		geometry_threshold: float | None = None,
		geometry_weight: float | None = None,
	) -> torch.Tensor | None:
		if not torch.is_tensor(pred_atom):
			raise TypeError("pred_atom must be a torch.Tensor.")

		ref_coords = torch.as_tensor(ref_coords, device=pred_atom.device, dtype=pred_atom.dtype)
		mask = _valid_mask(ref_coords)
		if mask.sum() == 0:
			return None

		pred_masked = pred_atom[mask]
		ref_masked = ref_coords[mask]

		if coord_scaler is not None:
			mean_t = torch.as_tensor(coord_scaler.mean_, device=pred_atom.device, dtype=pred_atom.dtype)
			scale_t = torch.as_tensor(coord_scaler.scale_, device=pred_atom.device, dtype=pred_atom.dtype)
			pred_scaled = (pred_masked - mean_t) / scale_t
			ref_scaled = (ref_masked - mean_t) / scale_t
			pred_geo = pred_masked
		else:
			pred_scaled = pred_masked
			ref_scaled = ref_masked
			pred_geo = pred_masked

		mse_loss = torch.mean((pred_scaled - ref_scaled) ** 2)

		if pred_geo.shape[0] > 1:
			threshold = self.geometry_distance_threshold if geometry_threshold is None else geometry_threshold
			weight = self.geometry_loss_weight if geometry_weight is None else geometry_weight
			step_dist = torch.linalg.norm(pred_geo[1:] - pred_geo[:-1], dim=-1)
			geometry_loss = torch.mean(F.relu(step_dist - threshold) ** 2)
			return mse_loss + weight * geometry_loss

		return mse_loss
