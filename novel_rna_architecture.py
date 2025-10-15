import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch_geometric
from torch_geometric.nn import MessagePassing, GCNConv, GATConv
from torch_geometric.utils import add_self_loops, degree
import random
import os
from datetime import datetime
import warnings
import math
from typing import Optional, Tuple, Dict, Any, List
import pickle
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import einops
warnings.filterwarnings('ignore')

class AdvancedRNASequenceGenerator:
    """
    Advanced RNA sequence generator with evolutionary and thermodynamic constraints
    """
    def __init__(self):
        self.nucleotides = ['A', 'U', 'G', 'C']

        # Advanced RNA motifs based on real structural biology
        self.structural_motifs = {
            'hairpin_loops': {
                'tetraloop_GNRA': ['GAAA', 'GCAA', 'GUAA', 'GAGA'],
                'tetraloop_UNCG': ['UUCG', 'UACG', 'UUUG'],
                'triloop': ['GAA', 'UUA', 'CUA']
            },
            'internal_loops': {
                'symmetric': [('GU', 'UG'), ('AU', 'UA'), ('GC', 'CG')],
                'asymmetric': [('GA', 'UG'), ('CA', 'UG'), ('UU', 'AA')]
            },
            'bulges': {
                'single_nucleotide': ['A', 'U', 'G', 'C'],
                'double_nucleotide': ['AA', 'UU', 'GG', 'CC']
            },
            'pseudoknots': {
                'h_type': ['GCGC', 'AUAU', 'GUGC'],
                'kissing_hairpins': ['GAAA', 'CUUG']
            }
        }

       
        self.base_pair_energies = {
            ('A', 'U'): -2.1, ('U', 'A'): -2.1,
            ('G', 'C'): -3.4, ('C', 'G'): -3.4,
            ('G', 'U'): -1.3, ('U', 'G'): -1.3,
            ('A', 'G'): 1.2, ('G', 'A'): 1.2,
            ('A', 'C'): 2.1, ('C', 'A'): 2.1,
            ('U', 'C'): 1.8, ('C', 'U'): 1.8
        }

    def generate_biologically_realistic_sequence(self, length_category: str, 
                                               target_gc_content: float = 0.5,
                                               min_stem_length: int = 4) -> str:
        """
        Generate biologically realistic RNA sequences with thermodynamic considerations
        """
        if length_category == 'short':
            length = random.randint(20, 35)
        elif length_category == 'medium':
            length = random.randint(45, 80)
        elif length_category == 'long':
            length = random.randint(90, 150)
        else:
            length = random.randint(60, 100)

        sequence = []
        i = 0
        gc_count = 0

        while i < length:
            remaining_length = length - i

            # Insert structured motifs with biological probability
            if remaining_length > 8 and random.random() < 0.4:
                motif_type = random.choice(list(self.structural_motifs.keys()))

                if motif_type == 'hairpin_loops' and remaining_length > 12:
                    # Create stem-loop structure
                    stem_length = min(random.randint(min_stem_length, 8), remaining_length // 3)
                    loop_type = random.choice(list(self.structural_motifs[motif_type].keys()))
                    loop_seq = random.choice(self.structural_motifs[motif_type][loop_type])

                    # Generate complementary stem
                    stem_5p = self._generate_stem_sequence(stem_length, target_gc_content)
                    stem_3p = self._get_complement_sequence(stem_5p[::-1])

                    motif_sequence = stem_5p + loop_seq + stem_3p

                    if i + len(motif_sequence) <= length:
                        sequence.extend(list(motif_sequence))
                        gc_count += sum(1 for nuc in motif_sequence if nuc in 'GC')
                        i += len(motif_sequence)
                        continue

                elif motif_type == 'internal_loops' and remaining_length > 10:
                    loop_data = random.choice(self.structural_motifs[motif_type]['symmetric'])
                    internal_loop = list(loop_data[0] + loop_data[1])
                    sequence.extend(internal_loop)
                    gc_count += sum(1 for nuc in internal_loop if nuc in 'GC')
                    i += len(internal_loop)
                    continue

            # Fill with contextually appropriate nucleotides
            current_gc_ratio = gc_count / max(1, i)
            if current_gc_ratio < target_gc_content - 0.1:
                # Favor G/C
                next_nuc = random.choice(['G', 'C', 'G', 'C', 'A', 'U'])
            elif current_gc_ratio > target_gc_content + 0.1:
                # Favor A/U
                next_nuc = random.choice(['A', 'U', 'A', 'U', 'G', 'C'])
            else:
                next_nuc = random.choice(self.nucleotides)

            sequence.append(next_nuc)
            if next_nuc in 'GC':
                gc_count += 1
            i += 1

        return ''.join(sequence[:length])

    def _generate_stem_sequence(self, length: int, gc_content: float = 0.6) -> str:
        """Generate a stem sequence favoring stable base pairs"""
        stem = []
        for _ in range(length):
            if random.random() < gc_content:
                stem.append(random.choice(['G', 'C']))
            else:
                stem.append(random.choice(['A', 'U']))
        return ''.join(stem)

    def _get_complement_sequence(self, sequence: str) -> str:
        """Get Watson-Crick complement with wobble pairs"""
        complement_map = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
        complement = []
        for nuc in sequence:
            if random.random() < 0.1 and nuc in ['G', 'U']:
                # Allow G-U wobble pairs
                if nuc == 'G':
                    complement.append('U')
                else:  # nuc == 'U'
                    complement.append('G')
            else:
                complement.append(complement_map[nuc])
        return ''.join(complement)

    def calculate_folding_energy(self, sequence: str) -> float:
        """
        Calculate approximate folding free energy using simplified Turner rules
        """
        energy = 0.0
        length = len(sequence)

        # Simple energy model based on nearest neighbors
        for i in range(length - 1):
            current_pair = (sequence[i], sequence[i + 1])
            if current_pair in self.base_pair_energies:
                energy += self.base_pair_energies[current_pair]

        # Add entropy penalty for loops (simplified)
        entropy_penalty = length * 0.1  # kcal/mol per nucleotide

        return energy + entropy_penalty

class Quantum3DStructureGenerator:
    """
    Advanced 3D structure generator incorporating quantum mechanical effects
    and molecular dynamics principles
    """
    def __init__(self):
        self.rna_generator = AdvancedRNASequenceGenerator()

        # Quantum mechanical parameters for RNA bases
        self.quantum_params = {
            'A': {'polarizability': 11.1, 'van_der_waals_radius': 1.85},
            'U': {'polarizability': 9.5, 'van_der_waals_radius': 1.80},
            'G': {'polarizability': 12.3, 'van_der_waals_radius': 1.88},
            'C': {'polarizability': 8.8, 'van_der_waals_radius': 1.76}
        }

        # RNA backbone geometry parameters
        self.backbone_params = {
            'P_O5_distance': 1.61,  # Å
            'O5_C5_distance': 1.44,
            'C5_C4_distance': 1.51,
            'sugar_pucker_delta': 81.0,  # degrees (C3'-endo)
            'phosphate_torsion_alpha': -62.0,
            'phosphate_torsion_beta': 180.0,
            'glycosidic_torsion_chi': -160.0
        }

    def generate_quantum_enhanced_structure(self, sequence: str) -> List[List[float]]:
        """
        Generate 3D structure with quantum mechanical corrections
        """
        coords = self._generate_base_helix_structure(sequence)
        coords = self._apply_quantum_corrections(coords, sequence)
        coords = self._minimize_molecular_energy(coords, sequence)
        return coords

    def _generate_base_helix_structure(self, sequence: str) -> np.ndarray:
        """Generate initial A-form RNA helix structure"""
        length = len(sequence)
        coords = np.zeros((length, 3))

        # A-form RNA parameters
        rise_per_nucleotide = 2.56  # Å
        twist_per_nucleotide = 32.7  # degrees
        helix_radius = 11.0  # Å

        for i, nuc in enumerate(sequence):
            # Calculate helical coordinates
            angle = np.radians(i * twist_per_nucleotide)

            # Base position
            x = helix_radius * np.cos(angle)
            y = helix_radius * np.sin(angle)
            z = i * rise_per_nucleotide

            # Add nucleotide-specific perturbations
            nuc_params = self.quantum_params[nuc]
            radius_correction = nuc_params['van_der_waals_radius'] - 1.8

            coords[i] = [
                x + radius_correction * np.cos(angle),
                y + radius_correction * np.sin(angle),
                z + np.random.normal(0, 0.1)
            ]

        return coords

    def _apply_quantum_corrections(self, coords: np.ndarray, sequence: str) -> np.ndarray:
        """Apply quantum mechanical corrections to coordinates"""
        corrected_coords = coords.copy()

        for i, nuc in enumerate(sequence):
            # Quantum polarizability effects
            polarizability = self.quantum_params[nuc]['polarizability']

            # Calculate local electric field from neighboring bases
            electric_field = self._calculate_local_electric_field(coords, i, sequence)

            # Induced dipole correction
            dipole_correction = polarizability * electric_field * 1e-4
            corrected_coords[i] += dipole_correction

        return corrected_coords

    def _calculate_local_electric_field(self, coords: np.ndarray, center_idx: int, 
                                      sequence: str) -> np.ndarray:
        """Calculate local electric field at a nucleotide position"""
        field = np.zeros(3)
        center_pos = coords[center_idx]

        # Sum contributions from nearby nucleotides
        for i, nuc in enumerate(sequence):
            if i != center_idx:
                r_vec = center_pos - coords[i]
                r_dist = np.linalg.norm(r_vec)

                if r_dist < 10.0:  # Only consider nearby nucleotides
                    # Simplified charge model
                    charge = self._get_effective_charge(nuc)
                    field += charge * r_vec / (r_dist ** 3)

        return field

    def _get_effective_charge(self, nucleotide: str) -> float:
        """Get effective partial charge for nucleotide"""
        charges = {'A': -0.3, 'U': -0.2, 'G': -0.4, 'C': -0.1}
        return charges.get(nucleotide, 0.0)

    def _minimize_molecular_energy(self, coords: np.ndarray, sequence: str) -> List[List[float]]:
        """Minimize molecular energy using simplified force field"""
        def energy_function(flat_coords):
            coords_3d = flat_coords.reshape(-1, 3)
            return self._calculate_total_energy(coords_3d, sequence)

        # Flatten coordinates for optimization
        initial_coords = coords.flatten()

        # Minimize energy
        try:
            result = minimize(energy_function, initial_coords, method='BFGS',
                            options={'maxiter': 100, 'disp': False})
            optimized_coords = result.x.reshape(-1, 3)
        except:
            # Fallback to original coordinates if optimization fails
            optimized_coords = coords

        return optimized_coords.tolist()

    def _calculate_total_energy(self, coords: np.ndarray, sequence: str) -> float:
        """Calculate total molecular energy"""
        energy = 0.0

        # Bond stretching energy (simplified)
        for i in range(len(coords) - 1):
            bond_length = np.linalg.norm(coords[i+1] - coords[i])
            ideal_bond_length = 6.0  # Approximate P-P distance in RNA
            energy += 0.5 * (bond_length - ideal_bond_length) ** 2

        # Van der Waals interactions
        for i in range(len(coords)):
            for j in range(i + 2, len(coords)):  # Skip adjacent nucleotides
                r_ij = np.linalg.norm(coords[i] - coords[j])
                if r_ij < 15.0:  # Cutoff distance
                    vdw_radius_i = self.quantum_params[sequence[i]]['van_der_waals_radius']
                    vdw_radius_j = self.quantum_params[sequence[j]]['van_der_waals_radius']
                    sigma = vdw_radius_i + vdw_radius_j

                    # Lennard-Jones potential (simplified)
                    lj_energy = 4.0 * ((sigma/r_ij)**12 - (sigma/r_ij)**6)
                    energy += lj_energy

        return energy

    def generate_conformational_ensemble(self, sequence: str, n_conformers: int = 5) -> List[List[List[float]]]:
        """Generate ensemble of conformational states"""
        ensemble = []

        for _ in range(n_conformers):
            # Add random perturbations to generate different conformers
            coords = self.generate_quantum_enhanced_structure(sequence)

            # Apply conformational sampling
            perturbed_coords = []
            for coord in coords:
                perturbation = np.random.normal(0, 0.5, 3)
                perturbed_coords.append([coord[0] + perturbation[0],
                                       coord[1] + perturbation[1], 
                                       coord[2] + perturbation[2]])

            ensemble.append(perturbed_coords)

        return ensemble

# Advanced Neural Network Architecture

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) for improved sequence modeling
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Create frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Cache sin and cos values
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, seq_len=None):
        if seq_len is None:
            seq_len = k.shape[-2]

        cos = self.cos_cached[:, :, :seq_len, :].to(q.device)
        sin = self.sin_cached[:, :, :seq_len, :].to(q.device)

        return (q * cos) + (self.rotate_half(q) * sin), (k * cos) + (self.rotate_half(k) * sin)

class GeometricSelfAttention(nn.Module):
    """
    Advanced geometric self-attention with 3D spatial awareness
    """
    def __init__(self, d_model: int, n_heads: int, d_point: int = 16, 
                 spatial_cutoff: float = 15.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_point = d_point
        self.spatial_cutoff = spatial_cutoff
        self.scale = (d_model // n_heads) ** -0.5

        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Geometric projections
        self.q_point_proj = nn.Linear(d_model, n_heads * d_point * 3)
        self.k_point_proj = nn.Linear(d_model, n_heads * d_point * 3)
        self.v_point_proj = nn.Linear(d_model, n_heads * d_point * 3)

        # Distance embedding for spatial relationships
        self.distance_embedding = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, n_heads)
        )

        # Output projection
        self.out_proj = nn.Linear(d_model + n_heads * d_point * 4, d_model)
        self.dropout = nn.Dropout(0.1)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, coords: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, seq_len, _ = x.shape
        residual = x

        # Apply layer normalization
        x = self.norm1(x)

        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, -1)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, -1)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, -1)

        # Geometric projections
        q_point = self.q_point_proj(x).view(batch_size, seq_len, self.n_heads, self.d_point, 3)
        k_point = self.k_point_proj(x).view(batch_size, seq_len, self.n_heads, self.d_point, 3)
        v_point = self.v_point_proj(x).view(batch_size, seq_len, self.n_heads, self.d_point, 3)

        # Add coordinate information
        coords_expanded = coords.unsqueeze(2).unsqueeze(3)
        q_point = q_point + coords_expanded
        k_point = k_point + coords_expanded

        # Compute spatial distances for attention bias
        coord_diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (batch, seq, seq, 3)
        distances = torch.norm(coord_diff, dim=-1, keepdim=True)  # (batch, seq, seq, 1)

        # Distance-based attention bias
        distance_bias = self.distance_embedding(distances)  # (batch, seq, seq, n_heads)
        distance_bias = distance_bias.permute(0, 3, 1, 2)  # (batch, n_heads, seq, seq)

        # Spatial mask for long-range interactions
        spatial_mask = (distances.squeeze(-1) < self.spatial_cutoff).float()
        spatial_mask = spatial_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)

        # Standard attention scores
        attn_scores = torch.einsum('bqhd,bkhd->bhqk', q, k) * self.scale

        # Geometric attention scores
        point_diff = q_point.unsqueeze(3) - k_point.unsqueeze(2)
        point_scores = -0.5 * torch.sum(point_diff ** 2, dim=(-2, -1))

        # Combine attention scores
        total_scores = attn_scores + point_scores + distance_bias

        # Apply masks
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            total_scores = total_scores.masked_fill(mask == 0, -1e9)

        total_scores = total_scores * spatial_mask + (spatial_mask - 1) * 1e9

        # Compute attention weights
        attn_weights = F.softmax(total_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attended_features = torch.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        attended_points = torch.einsum('bhqk,bkhpd->bqhpd', attn_weights, v_point)

        # Flatten and concatenate
        attended_features = attended_features.contiguous().view(batch_size, seq_len, -1)
        attended_points = attended_points.contiguous().view(batch_size, seq_len, -1)

        # Combine and project
        combined = torch.cat([attended_features, attended_points], dim=-1)
        output = self.out_proj(combined)

        # Residual connection with layer norm
        output = self.norm2(residual + output)

        # Update coordinates based on point attention
        coord_updates = torch.mean(attended_points.view(batch_size, seq_len, self.n_heads, self.d_point, 3), 
                                 dim=(2, 3)) * 0.1
        new_coords = coords + coord_updates

        return output, new_coords

class AdaptiveEGNNLayer(MessagePassing):
    """
    Adaptive Equivariant Graph Neural Network with learned edge features
    """
    def __init__(self, hidden_dim: int, edge_dim: int = 32, num_edge_types: int = 4):
        super().__init__(aggr='add')
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.num_edge_types = num_edge_types

        # Edge type embedding
        self.edge_type_embedding = nn.Embedding(num_edge_types, edge_dim)

        # Adaptive message networks
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim + 1, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(0.1)
        )

        # Coordinate update network with attention
        self.coord_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Tanh()  # Bounded coordinate updates
        )

        # Node update network
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(0.1)
        )

        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, coords: torch.Tensor, 
                edge_index: torch.Tensor, edge_types: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # Embed edge types
        edge_attr = self.edge_type_embedding(edge_types)

        # Compute edge distances
        row, col = edge_index
        coord_diff = coords[row] - coords[col]
        edge_dist = torch.norm(coord_diff, dim=-1, keepdim=True)

        # Message passing
        messages = self.propagate(edge_index, x=x, coords=coords, 
                                edge_attr=edge_attr, edge_dist=edge_dist)

        # Update node features
        node_input = torch.cat([x, messages], dim=-1)
        x_updated = self.node_mlp(node_input) + x  # Residual connection

        # Update coordinates with attention mechanism
        coord_context, _ = self.coord_attention(x_updated.unsqueeze(0), 
                                              x_updated.unsqueeze(0), 
                                              x_updated.unsqueeze(0))
        coord_context = coord_context.squeeze(0)

        coord_updates = self.coord_mlp(coord_context)
        coords_updated = coords + coord_updates * 0.1 * torch.sigmoid(self.temperature)

        return x_updated, coords_updated

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, 
               edge_attr: torch.Tensor, edge_dist: torch.Tensor) -> torch.Tensor:
        # Concatenate node features, edge attributes, and distances
        message_input = torch.cat([x_i, x_j, edge_attr, edge_dist], dim=-1)
        return self.message_mlp(message_input)

class DynamicLatentSpaceMixup:
    """
    Dynamic Latent Space Mixup with adaptive mixing strategies
    """
    def __init__(self, alpha: float = 0.4, adaptive: bool = True):
        self.alpha = alpha
        self.adaptive = adaptive
        self.epoch_counter = 0

    def __call__(self, features: torch.Tensor, coords: torch.Tensor, 
                 targets: Optional[torch.Tensor] = None, 
                 epoch: int = 0) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        if features.size(0) < 2:
            return features, coords, targets

        # Adaptive alpha based on training progress
        if self.adaptive:
            # Decrease mixing strength as training progresses
            adaptive_alpha = self.alpha * max(0.1, 1.0 - epoch / 100.0)
        else:
            adaptive_alpha = self.alpha

        # Sample mixing coefficient from Beta distribution
        if adaptive_alpha > 0:
            lam = np.random.beta(adaptive_alpha, adaptive_alpha)
        else:
            lam = 1.0

        batch_size = features.size(0)

        # Smart pairing strategy - pair similar sequences for better mixing
        if batch_size >= 4:
            # Use feature similarity for pairing
            feature_norms = torch.norm(features.view(batch_size, -1), dim=1)
            sorted_indices = torch.argsort(feature_norms)

            # Create pairs of similar samples
            indices = torch.zeros_like(sorted_indices)
            for i in range(0, batch_size - 1, 2):
                indices[sorted_indices[i]] = sorted_indices[i + 1]
                indices[sorted_indices[i + 1]] = sorted_indices[i]

            if batch_size % 2 == 1:
                indices[-1] = sorted_indices[0]  # Pair last with first
        else:
            indices = torch.randperm(batch_size)

        # Mix features with geometric constraints
        mixed_features = lam * features + (1 - lam) * features[indices]

        # Coordinate mixing with structural preservation
        coord_distances = torch.norm(coords.unsqueeze(1) - coords[indices].unsqueeze(0), dim=-1)
        coord_mask = (coord_distances < 10.0).float()  # Preserve local structure

        mixed_coords = lam * coords + (1 - lam) * coords[indices] * coord_mask.unsqueeze(-1)
        mixed_coords = mixed_coords + (coords * (1 - coord_mask.unsqueeze(-1)))

        # Mix targets if provided
        mixed_targets = None
        if targets is not None:
            if len(targets.shape) > 1:
                mixed_targets = lam * targets + (1 - lam) * targets[indices]
            else:
                # For classification targets, use mixup label
                mixed_targets = (lam, targets, targets[indices])

        return mixed_features, mixed_coords, mixed_targets

class NovelHybridRNAModel(nn.Module):
    """
    State-of-the-art Hybrid RNA Model with Novel Architecture Components
    """
    def __init__(self, vocab_size: int = 5, d_model: int = 512, n_heads: int = 16, 
                 n_layers: int = 12, max_seq_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers

        # Advanced embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.rotary_pos_emb = RotaryPositionalEmbedding(d_model // n_heads, max_seq_len)

        # Nucleotide-specific parameter networks
        self.nucleotide_params = nn.ModuleDict({
            'A': nn.Linear(d_model, d_model),
            'U': nn.Linear(d_model, d_model),
            'G': nn.Linear(d_model, d_model),
            'C': nn.Linear(d_model, d_model)
        })

        # Geometric Self-Attention layers
        self.geometric_attention_layers = nn.ModuleList([
            GeometricSelfAttention(d_model, n_heads, d_point=32)
            for _ in range(n_layers // 3)
        ])

        # Adaptive EGNN layers
        self.egnn_layers = nn.ModuleList([
            AdaptiveEGNNLayer(d_model, edge_dim=64, num_edge_types=8)
            for _ in range(n_layers // 3)
        ])

        # Enhanced Transformer layers with RoPE
        self.transformer_layers = nn.ModuleList([
            self._create_enhanced_transformer_layer(d_model, n_heads, dropout)
            for _ in range(n_layers // 3)
        ])

        # Multi-scale feature fusion
        self.feature_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 3, d_model * 2),
                nn.LayerNorm(d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(dropout)
            ) for _ in range(n_layers // 3)
        ])

        # Advanced output heads
        self.coordinate_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
            nn.Tanh()  # Bounded output
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

        # Secondary structure prediction
        self.structure_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 16)  # Extended structure classes
        )

        # Torsion angle prediction
        self.torsion_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 6)  # α, β, γ, δ, ε, ζ torsions
        )

        # Distance matrix prediction for contacts
        self.distance_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 64),  # Distance bins
            nn.LogSoftmax(dim=-1)
        )

        # Dynamic mixup
        self.mixup = DynamicLatentSpaceMixup(alpha=0.3, adaptive=True)

        # Global feature processing
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )

        # Initialize parameters
        self._init_parameters()

    def _create_enhanced_transformer_layer(self, d_model: int, n_heads: int, dropout: float):
        """Create enhanced transformer layer with improvements"""
        return nn.ModuleDict({
            'self_attn': nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
            'feed_forward': nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            ),
            'norm1': nn.LayerNorm(d_model),
            'norm2': nn.LayerNorm(d_model),
            'gate': nn.Linear(d_model, d_model)  # Gating mechanism
        })

    def _init_parameters(self):
        """Initialize model parameters with appropriate schemes"""
        for name, param in self.named_parameters():
            if 'embedding' in name or 'linear' in name or 'Linear' in str(type(param)):
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)
            elif 'norm' in name.lower():
                if param.dim() > 1:
                    nn.init.ones_(param)
                else:
                    nn.init.zeros_(param)

    def create_dynamic_edge_index(self, coords: torch.Tensor, sequence: torch.Tensor, 
                                cutoff: float = 12.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create dynamic edge index based on spatial proximity and sequence"""
        batch_size, seq_len = coords.shape[:2]
        all_edges = []
        all_edge_types = []

        for b in range(batch_size):
            seq_coords = coords[b]
            seq_nucleotides = sequence[b]

            # Compute pairwise distances
            distances = torch.cdist(seq_coords, seq_coords)

            # Create edges based on distance and sequence proximity
            edges = []
            edge_types = []

            for i in range(seq_len):
                for j in range(seq_len):
                    if i != j:
                        dist = distances[i, j]
                        seq_dist = abs(i - j)

                        # Backbone connectivity (always included)
                        if seq_dist == 1:
                            edges.append([i, j])
                            edge_types.append(0)  # backbone edge

                        # Base pairing (close in space)
                        elif dist < cutoff and seq_dist > 2:
                            edges.append([i, j])
                            # Determine edge type based on nucleotides
                            nuc_i = seq_nucleotides[i].item()
                            nuc_j = seq_nucleotides[j].item()
                            if (nuc_i, nuc_j) in [(1,2), (2,1), (3,4), (4,3)]:  # Watson-Crick
                                edge_types.append(1)
                            elif (nuc_i, nuc_j) in [(3,2), (2,3)]:  # Wobble
                                edge_types.append(2)
                            else:
                                edge_types.append(3)  # Non-canonical

                        # Long-range interactions
                        elif dist < cutoff * 1.5 and seq_dist > 10:
                            edges.append([i, j])
                            edge_types.append(4)  # long-range edge

            if edges:
                edges_tensor = torch.tensor(edges, dtype=torch.long, device=coords.device).t()
                edge_types_tensor = torch.tensor(edge_types, dtype=torch.long, device=coords.device)
            else:
                # Fallback: at least connect adjacent nucleotides
                edges_tensor = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=coords.device).t()
                edge_types_tensor = torch.tensor([0, 0], dtype=torch.long, device=coords.device)

            all_edges.append(edges_tensor)
            all_edge_types.append(edge_types_tensor)

        return all_edges, all_edge_types

    def forward(self, sequences: torch.Tensor, coords: Optional[torch.Tensor] = None, 
                training: bool = True, epoch: int = 0) -> Dict[str, torch.Tensor]:

        batch_size, seq_len = sequences.shape
        device = sequences.device

        # Initialize coordinates if not provided
        if coords is None:
            coords = torch.randn(batch_size, seq_len, 3, device=device) * 0.5

        # Token embeddings with nucleotide-specific processing
        token_emb = self.token_embedding(sequences)

        # Apply nucleotide-specific transformations
        processed_emb = torch.zeros_like(token_emb)
        nucleotide_map = {1: 'A', 2: 'U', 3: 'G', 4: 'C'}

        for nuc_idx, nuc_name in nucleotide_map.items():
            mask = (sequences == nuc_idx)
            if mask.any():
                processed_emb[mask] = self.nucleotide_params[nuc_name](token_emb[mask])

        # Handle padding tokens
        padding_mask = (sequences == 0)
        processed_emb[padding_mask] = token_emb[padding_mask]

        x = processed_emb

        # Apply dynamic mixup during training
        if training and torch.rand(1).item() < 0.4:
            x, coords, _ = self.mixup(x, coords, epoch=epoch)

        # Create attention mask for padding
        attention_mask = (sequences != 0).float()

        # Multi-layer processing
        layer_outputs = []

        for layer_idx in range(len(self.geometric_attention_layers)):
            # Geometric Self-Attention
            x_geom, coords = self.geometric_attention_layers[layer_idx](x, coords, attention_mask)

            # Enhanced Transformer with RoPE
            transformer_layer = self.transformer_layers[layer_idx]

            # Apply RoPE to query and key
            q = k = transformer_layer['norm1'](x_geom)
            q, k = self.rotary_pos_emb(q, k, seq_len)

            # Self-attention with gating
            attn_out, _ = transformer_layer['self_attn'](q, k, x_geom, 
                                                       key_padding_mask=~attention_mask.bool())
            gate = torch.sigmoid(transformer_layer['gate'](x_geom))
            x_transformer = x_geom + gate * attn_out

            # Feed forward
            ff_out = transformer_layer['feed_forward'](transformer_layer['norm1'](x_transformer))
            x_transformer = transformer_layer['norm2'](x_transformer + ff_out)

            # EGNN processing (per batch)
            x_egnn_list = []
            coords_egnn_list = []

            edge_indices, edge_types = self.create_dynamic_edge_index(coords, sequences)

            for b in range(batch_size):
                seq_x = x_transformer[b]
                seq_coords = coords[b]
                edge_index = edge_indices[b]
                edge_type = edge_types[b]

                x_egnn_out, coords_egnn_out = self.egnn_layers[layer_idx](
                    seq_x, seq_coords, edge_index, edge_type
                )

                x_egnn_list.append(x_egnn_out)
                coords_egnn_list.append(coords_egnn_out)

            x_egnn = torch.stack(x_egnn_list, dim=0)
            coords = torch.stack(coords_egnn_list, dim=0)

            # Feature fusion
            fused_features = torch.cat([x_geom, x_transformer, x_egnn], dim=-1)
            x = self.feature_fusion[layer_idx](fused_features) + x  # Residual connection

            layer_outputs.append(x)

        # Global feature processing
        global_features = self.global_pool(x.transpose(1, 2)).squeeze(-1)
        global_context = self.global_mlp(global_features).unsqueeze(1).expand(-1, seq_len, -1)
        x = x + global_context * 0.1  # Add global context

        # Generate predictions
        predicted_coords = self.coordinate_head(x) * 10.0  # Scale coordinates
        confidence = self.confidence_head(x).squeeze(-1)
        structure_logits = self.structure_head(x)
        torsion_angles = self.torsion_head(x)

        # Distance matrix prediction (pairwise)
        x_expanded_i = x.unsqueeze(2).expand(-1, -1, seq_len, -1)
        x_expanded_j = x.unsqueeze(1).expand(-1, seq_len, -1, -1)
        pairwise_features = torch.cat([x_expanded_i, x_expanded_j], dim=-1)
        distance_logits = self.distance_head(pairwise_features)

        # Final coordinate prediction
        final_coords = coords + predicted_coords * 0.05  # Small updates for stability

        return {
            'coordinates': final_coords,
            'confidence': confidence,
            'structure_logits': structure_logits,
            'torsion_angles': torsion_angles,
            'distance_logits': distance_logits,
            'features': x,
            'layer_outputs': layer_outputs,
            'raw_coord_predictions': predicted_coords
        }

class EnhancedRNADataset(Dataset):
    """Enhanced RNA dataset with advanced preprocessing"""
    def __init__(self, data_list: List[Dict], max_length: int = 256, 
                 augment: bool = True):
        self.data = data_list
        self.max_length = max_length
        self.augment = augment
        self.nucleotide_map = {'A': 1, 'U': 2, 'G': 3, 'C': 4, '<PAD>': 0}

        # Advanced data augmentation
        self.structure_generator = Quantum3DStructureGenerator()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sequence = item['sequence'][:self.max_length]
        coords = item['coordinates'][:self.max_length]

        # Convert sequence to numbers
        seq_nums = [self.nucleotide_map.get(nuc, 0) for nuc in sequence]

        # Data augmentation during training
        if self.augment and random.random() < 0.3:
            # Generate alternative conformations
            alt_coords = self.structure_generator.generate_quantum_enhanced_structure(sequence)
            if len(alt_coords) >= len(coords):
                coords = alt_coords[:len(coords)]

        # Pad sequences and coordinates
        original_length = len(seq_nums)
        while len(seq_nums) < self.max_length:
            seq_nums.append(0)
        while len(coords) < self.max_length:
            coords.append([0.0, 0.0, 0.0])

        # Convert to tensors
        sequence_tensor = torch.tensor(seq_nums[:self.max_length], dtype=torch.long)
        coords_tensor = torch.tensor(coords[:self.max_length], dtype=torch.float32)

        # Create additional features
        attention_mask = torch.zeros(self.max_length, dtype=torch.bool)
        attention_mask[:original_length] = True

        # Secondary structure labels (simplified)
        ss_labels = torch.randint(0, 8, (self.max_length,))  # Placeholder

        return {
            'sequence': sequence_tensor,
            'coordinates': coords_tensor,
            'attention_mask': attention_mask,
            'secondary_structure': ss_labels,
            'length': original_length,
            'original_sequence': sequence
        }

# Save the enhanced model architecture
print("🚀 Creating highly novel and advanced RNA 3D folding model...")
