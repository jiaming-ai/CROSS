from .boq import BoQ
import numpy as np
import torch
from cross.core.types import Atlas, Keyframe
from cross.utils.profile import timeit
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

from cross.core.config import RetrievalConfig, VPRModelType


class KeyframeDatabase:
    def __init__(
        self,
        system,
        device: str = "cuda",
        config: Union[RetrievalConfig, None] = None,
    ):
        """Database for posed RGBD with efficient embedding management.

        Args:
            system: System
            device: Device to store tensors
            config: RetrievalConfig with VPR model type, buffer size, and query parameters.
        """
        cfg = config or RetrievalConfig()
        self.system = system
        self.device = device

        # Atlas management
        self._atlases: Dict[int, Atlas] = {}  # Maps atlas_id to Atlas object
        self._next_atlas_id: int = 0

        # Store PosedRGBD objects by atlas
        self._keyframe_by_atlas: Dict[Atlas, List[Keyframe]] = defaultdict(list)

        # Index management
        self._atlas_to_indices: Dict[Atlas, List[int]] = defaultdict(list)  # Maps atlas to list of buffer indices
        self._index_to_atlas_idx: Dict[int, Tuple[Atlas, int]] = {}  # Maps buffer index to (atlas, list_idx)

        # VPR model setup
        self.vpr_model_type = cfg.vpr_model_type
        if self.vpr_model_type == VPRModelType.BOQ:
            self.vpr_model = BoQ(backbone_name="resnet50", device=device)
        else:
            raise ValueError(f"VPR model {self.vpr_model_type} not supported")

        # Embedding management
        self._initial_buffer_size = cfg.initial_buffer_size
        self._embedding_buffer = torch.zeros(
            (self._initial_buffer_size, self.vpr_model.get_embed_dim()),  # Assuming 2048-dim embeddings
            device=self.device
        )
        self._current_size = 0

        # Query parameters
        self.score_threshold_high = cfg.vpr_score_threshold_high
        self.score_threshold_low = cfg.vpr_score_threshold_low
        self.top_k = cfg.top_k

    def _extend_buffer(self, min_size: int):
        """Extend the embedding buffer if needed."""
        if min_size <= self._embedding_buffer.shape[0]:
            return
            
        new_size = max(min_size, self._embedding_buffer.shape[0] * 2)
        new_buffer = torch.zeros(
            (new_size, self._embedding_buffer.shape[1]),
            device=self.device
        )
        new_buffer[:self._current_size] = self._embedding_buffer[:self._current_size]
        self._embedding_buffer = new_buffer

    def get_all_keyframes(self, atlas: Atlas= None) -> List[Keyframe]:
        """Get all keyframes from the database."""
        if atlas is None:
            # return all keyframes
            keyframes = []
            for atlas in self._keyframe_by_atlas:
                keyframes.extend(self._keyframe_by_atlas[atlas])
            return keyframes
        else:
            return self._keyframe_by_atlas[atlas]

    @timeit
    def insert(
        self, 
        id: int,
        raw_rgb_image: torch.Tensor,
        depth_image: torch.Tensor,
        mu: torch.Tensor = None,
        sigma: torch.Tensor = None,
        weights: torch.Tensor = None,
        timestamp: float = None,
        atlas: Atlas = None,
        temporary: bool = False,
        last_pgo_step: int = -1,
    ) -> Keyframe:
        """Insert a Keyframe into the database.

        Args:
            id: The id of the keyframe
            img_tensor: The image of the place, tensor of shape (3, H, W)
            atlas: The atlas of the place
            pose_in_atlas: The pose of the place in the atlas frame
            temporary: Whether the keyframe is temporary
            t: The timestamp of the place
            raw_rgb_image: The raw RGB image of the place, shape (H, W, 3) nd uint8
        """
        # Get embedding
        embedding = self.vpr_model.get_embedding(raw_rgb_image)
        
        # Create PosedRGBD object
        keyframe = Keyframe(
            raw_rgb_image=raw_rgb_image,
            depth_image=depth_image,
            pose_mu=mu,
            pose_std=sigma,
            pose_weights=weights,
            timestamp=timestamp,
            atlas=atlas,
            temporary=temporary,
            last_pgo_step=last_pgo_step,
        )
        
        # Add to storage
        list_idx = len(self._keyframe_by_atlas[atlas])
        self._keyframe_by_atlas[atlas].append(keyframe)
        
        # Ensure buffer has space, 
        # add 100 to the current size to avoid too many reallocations
        self._extend_buffer(self._current_size + 100)
        
        # Add embedding to buffer
        self._embedding_buffer[self._current_size] = embedding
        self._atlas_to_indices[atlas].append(self._current_size)
        self._index_to_atlas_idx[self._current_size] = (atlas, list_idx)
        self._current_size += 1

        return keyframe

    def remove(
        self, 
        idx: int,
        buffer_idx: int,
        atlas: Atlas = 0, 
    ):
        """Remove a Keyframe from the database.
        Args:
            atlas: The atlas containing the Keyframe
            idx: Index of the Keyframe in the atlas's list
        """
        if atlas not in self._keyframe_by_atlas or idx >= len(self._keyframe_by_atlas[atlas]):
            return
            
        # Find the buffer index for this place
        buffer_idx = None
        for i, (a, list_i) in self._index_to_atlas_idx.items():
            if a == atlas and list_i == idx:
                buffer_idx = i
                break
                
        if buffer_idx is None:
            return
            
        # Remove from storage
        self._keyframe_by_atlas[atlas].pop(idx)
        
        # Remove from indices
        self._atlas_to_indices[atlas].remove(buffer_idx)
        del self._index_to_atlas_idx[buffer_idx]
        
        # If this was the last element in the buffer, just decrease size
        if buffer_idx == self._current_size - 1:
            self._current_size -= 1
            return
            
        # Otherwise, move the last element to this position
        last_idx = self._current_size - 1
        self._embedding_buffer[buffer_idx] = self._embedding_buffer[last_idx]
        
        # Update indices for the moved element
        last_atlas, last_list_idx = self._index_to_atlas_idx[last_idx]
        self._atlas_to_indices[last_atlas].remove(last_idx)
        self._atlas_to_indices[last_atlas].append(buffer_idx)
        self._index_to_atlas_idx[buffer_idx] = (last_atlas, last_list_idx)
        del self._index_to_atlas_idx[last_idx]
        
        self._current_size -= 1

    def get_size(self):
        return self._current_size
    
    def create_atlas(self) -> Atlas:
        """Create a new atlas and return it."""
        atlas = Atlas(id=self._next_atlas_id)
        self._atlases[self._next_atlas_id] = atlas
        self._next_atlas_id += 1
        return atlas
    
    def get_atlas(self, atlas_id: int) -> Optional[Atlas]:
        """Get an atlas by ID."""
        return self._atlases.get(atlas_id)
    
    def get_all_atlases(self) -> List[Atlas]:
        """Get all atlases."""
        return list(self._atlases.values())
    
    def save_state(self):
        """Save the database state for map persistence.
        
        Returns:
            dict: Database state including keyframes, embeddings, and atlases
        """
        db_keyframes = []
        for atlas in self._keyframe_by_atlas:
            for kf in self._keyframe_by_atlas[atlas]:
                db_keyframes.append({
                    "id": kf.id,
                    "raw_rgb_image": kf.raw_rgb_image.cpu() if kf.raw_rgb_image is not None else None,
                    "depth_image": kf.depth_image.cpu() if kf.depth_image is not None else None,
                    "pose_mu": kf.pose_mu.cpu() if kf.pose_mu is not None else None,
                    "pose_std": kf.pose_std.cpu() if kf.pose_std is not None else None,
                    "pose_weights": kf.pose_weights.cpu() if kf.pose_weights is not None else None,
                    "atlas_id": kf.atlas.id if kf.atlas is not None else None,
                    "timestamp": kf.timestamp,
                    "temporary": kf.temporary,
                    "last_pgo_step": kf.last_pgo_step,
                })
        
        return {
            "keyframes": db_keyframes,
            "embeddings": self._embedding_buffer[:self._current_size].cpu(),
            "atlas_to_indices": {atlas.id: indices for atlas, indices in self._atlas_to_indices.items()},
            "index_to_atlas_idx": {buf_idx: (atlas.id, list_idx) for buf_idx, (atlas, list_idx) in self._index_to_atlas_idx.items()},
            "current_size": self._current_size,
            "atlases": {atlas_id: {"id": atlas.id} for atlas_id, atlas in self._atlases.items()},
            "next_atlas_id": self._next_atlas_id,
        }
    
    def load_state(self, db_data: dict, storage_device: str):
        """Load the database state from saved data.
        
        Args:
            db_data: Dictionary containing saved database state
            storage_device: Device to store tensors
            
        Returns:
            dict: Mapping from keyframe ID to Keyframe object
        """
        self._keyframe_by_atlas.clear()
        self._atlas_to_indices.clear()
        self._index_to_atlas_idx.clear()
        self._atlases.clear()
        
        # Restore atlases
        for atlas_id, atlas_data in db_data["atlases"].items():
            atlas = Atlas(id=atlas_data["id"])
            self._atlases[atlas_data["id"]] = atlas
        self._next_atlas_id = db_data["next_atlas_id"]
        
        # Restore embeddings
        embeddings = db_data["embeddings"].to(self.device)
        self._current_size = db_data["current_size"]
        
        # Extend buffer if needed
        self._extend_buffer(self._current_size)
        self._embedding_buffer[:self._current_size] = embeddings
        
        # Create a map to track all keyframes by ID
        all_keyframes_map = {}
        
        # Restore permanent keyframes from database
        for kf_data in db_data["keyframes"]:
            atlas = self._atlases[kf_data["atlas_id"]] if kf_data["atlas_id"] is not None else None
            
            # Create Keyframe object
            kf = Keyframe(
                raw_rgb_image=kf_data["raw_rgb_image"].to(storage_device) if kf_data["raw_rgb_image"] is not None else None,
                depth_image=kf_data["depth_image"].to(storage_device) if kf_data["depth_image"] is not None else None,
                pose_mu=kf_data["pose_mu"].to(storage_device) if kf_data["pose_mu"] is not None else None,
                pose_std=kf_data["pose_std"].to(storage_device) if kf_data["pose_std"] is not None else None,
                pose_weights=kf_data["pose_weights"].to(storage_device) if kf_data["pose_weights"] is not None else None,
                atlas=atlas,
                timestamp=kf_data["timestamp"],
                temporary=kf_data["temporary"],
                last_pgo_step=kf_data["last_pgo_step"],
            )

            # Manually set the ID to match the saved one
            kf.id = kf_data["id"]

            # Add to database storage
            list_idx = len(self._keyframe_by_atlas[atlas])
            self._keyframe_by_atlas[atlas].append(kf)
            
            all_keyframes_map[kf.id] = kf
        
        # Restore database index mappings
        for atlas_id, indices in db_data["atlas_to_indices"].items():
            atlas = self._atlases[int(atlas_id)]
            self._atlas_to_indices[atlas] = indices
        
        for buffer_idx, (atlas_id, list_idx) in db_data["index_to_atlas_idx"].items():
            atlas = self._atlases[atlas_id]
            self._index_to_atlas_idx[buffer_idx] = (atlas, list_idx)
        
        return all_keyframes_map
    
    @timeit
    def query(
        self, 
        img: torch.Tensor,
        target_atlases: Optional[List[Atlas]] = None,
    ) -> List[Tuple[float, Keyframe]]:
        """Query the database for the most likely Keyframe.

        Args:
            img: The image to query the place database with, shape (3, H, W)
            target_atlases: Optional list of atlases to restrict the search to

        Returns:
            List of (score, Keyframe) tuples in descending order of confidence
        """
        if self._current_size == 0:
            return {
                "scores": [],
                "keyframes": [],
            }

        # Get query embedding and move to correct device
        query_embedding = self.vpr_model.get_embedding(img)
        
        # Get relevant embeddings
        if target_atlases is not None:
            valid_indices = []
            for atlas in target_atlases:
                valid_indices.extend(self._atlas_to_indices[atlas])
            if not valid_indices:
                return {
                    "scores": [],
                    "keyframes": [],
                }
            database_emb = self._embedding_buffer[valid_indices]
        else:
            database_emb = self._embedding_buffer[:self._current_size]
        
        # Compute similarity scores
        scores = database_emb @ query_embedding.unsqueeze(-1)
        scores = scores.squeeze(-1)  # Only squeeze the last dimension to avoid making it 0-dimensional
        
        # Filter and sort scores
        # `scores` are similarity scores against `database_emb`
        
        # Find the indices in `scores` (and thus `database_emb`) that pass the threshold
        original_indices_passing_threshold = (scores > self.score_threshold_high).nonzero(as_tuple=True)[0]
        if original_indices_passing_threshold.numel() == 0:
            original_indices_passing_threshold = (scores > self.score_threshold_low).nonzero(as_tuple=True)[0]
        
        if original_indices_passing_threshold.numel() == 0:
            return {
                "scores": [],
                "keyframes": [],
            }
            
        # Get the actual scores for these candidates
        scores_of_candidates = scores[original_indices_passing_threshold]
        
        # Sort these candidate scores and get their relative indices (i.e., indices into scores_of_candidates)
        # These sorted_relative_indices will point to elements in scores_of_candidates (and by extension, original_indices_passing_threshold)
        # in descending order of score.
        sorted_relative_indices = scores_of_candidates.argsort(descending=True)
        
        # Select the top_k relative indices
        top_k_relative_indices = sorted_relative_indices[:self.top_k]
        
        # Use these top_k relative indices to get the actual top_k scores
        final_top_k_scores = scores_of_candidates[top_k_relative_indices]
        
        # And use them to get the top_k original indices (indices into `scores` or `database_emb`)
        final_top_k_original_db_indices = original_indices_passing_threshold[top_k_relative_indices]
        
        result_scores = []
        result_keyframes = []
        
        # Loop through the final top-k items
        for i in range(final_top_k_original_db_indices.numel()):
            # original_db_idx_in_scores is an index into database_emb
            original_db_idx_in_scores = final_top_k_original_db_indices[i] 
            score_value = final_top_k_scores[i].item()

            if target_atlases is not None:
                # original_db_idx_in_scores is an index for database_emb, which was self._embedding_buffer[valid_indices].
                # So, valid_indices[original_db_idx_in_scores.item()] gives the true buffer_idx.
                buffer_idx = valid_indices[original_db_idx_in_scores.item()]
            else:
                # original_db_idx_in_scores is an index for database_emb, which was self._embedding_buffer[:self._current_size].
                # So, original_db_idx_in_scores is directly the buffer_idx.
                buffer_idx = original_db_idx_in_scores.item()
            
            atlas, list_idx = self._index_to_atlas_idx[buffer_idx]
            result_scores.append(score_value)
            result_keyframes.append(self._keyframe_by_atlas[atlas][list_idx])
            
        return {
            "scores": result_scores,
            "keyframes": result_keyframes,
        }
    
    
