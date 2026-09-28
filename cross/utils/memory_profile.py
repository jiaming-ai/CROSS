import torch
import psutil
from loguru import logger
from typing import Dict


def get_tensor_memory(tensor: torch.Tensor) -> int:
    """Get memory usage of a tensor in bytes."""
    if tensor is None:
        return 0
    return tensor.element_size() * tensor.nelement()

def get_lietensor_memory(lietensor) -> int:
    """Get memory usage of a LieTensor in bytes."""
    if lietensor is None:
        return 0
    return get_tensor_memory(lietensor.tensor())
    
def get_memory_usage(system, detailed: bool = True) -> Dict:
    """Get detailed memory usage statistics for the system.
    
    Args:
        detailed: If True, return detailed breakdown; if False, only return totals
        
    Returns:
        Dictionary containing memory usage information in MB
    """
    def bytes_to_mb(bytes_val):
        return bytes_val / (1024 * 1024)
    
    memory_stats = {}
    
    # --- 1. Database Memory ---
    db_memory = {
        "total": 0,
        "keyframes": {
            "count": 0,
            "rgb_images": 0,
            "depth_images": 0,
            "pose_data": 0,
            "total": 0
        },
        "embeddings": 0,
        "vpr_model": 0,
    }
    
    # Count keyframes and their memory
    all_kfs = system.db.get_all_keyframes()
    db_memory["keyframes"]["count"] = len(all_kfs)
    
    for kf in all_kfs:
        if kf.raw_rgb_image is not None:
            db_memory["keyframes"]["rgb_images"] += get_tensor_memory(kf.raw_rgb_image)
        if kf.depth_image is not None:
            db_memory["keyframes"]["depth_images"] += get_tensor_memory(kf.depth_image)
        if kf.pose_mu is not None:
            db_memory["keyframes"]["pose_data"] += get_lietensor_memory(kf.pose_mu)
        if kf.pose_std is not None:
            db_memory["keyframes"]["pose_data"] += get_lietensor_memory(kf.pose_std)
        if kf.pose_weights is not None:
            db_memory["keyframes"]["pose_data"] += get_tensor_memory(kf.pose_weights)
    
    db_memory["keyframes"]["total"] = (
        db_memory["keyframes"]["rgb_images"] + 
        db_memory["keyframes"]["depth_images"] + 
        db_memory["keyframes"]["pose_data"]
    )
    
    # Embeddings
    if hasattr(system.db, '_embedding_buffer'):
        db_memory["embeddings"] = get_tensor_memory(system.db._embedding_buffer)
    
    # VPR model (approximate, includes model parameters)
    if hasattr(system.db, 'vpr_model') and hasattr(system.db.vpr_model, 'model'):
        for param in system.db.vpr_model.model.parameters():
            db_memory["vpr_model"] += get_tensor_memory(param)
    
    db_memory["total"] = (
        db_memory["keyframes"]["total"] + 
        db_memory["embeddings"] + 
        db_memory["vpr_model"]
    )
    
    memory_stats["database"] = db_memory
    
    # --- 2. Hypothesis Manager Memory ---
    hypo_memory = {
        "total": 0,
        "temporary_keyframes": {
            "count": 0,
            "rgb_images": 0,
            "depth_images": 0,
            "pose_data": 0,
            "total": 0
        },
        "edges": {
            "odom_edges_count": len(system.hypothesis_manager.odom_edges),
            "visual_edges_count": 0,
            "odom_edges_memory": 0,
            "visual_edges_memory": 0,
            "total": 0
        },
        "current_distribution": 0,
        "hypotheses_count": len(system.hypothesis_manager.hypotheses),
    }
    
    # Temporary keyframes (not in database)
    temp_kf_count = 0
    for kf in system.hypothesis_manager.nodes.values():
        if kf.temporary:
            temp_kf_count += 1
            if kf.raw_rgb_image is not None:
                hypo_memory["temporary_keyframes"]["rgb_images"] += get_tensor_memory(kf.raw_rgb_image)
            if kf.depth_image is not None:
                hypo_memory["temporary_keyframes"]["depth_images"] += get_tensor_memory(kf.depth_image)
            if kf.pose_mu is not None:
                hypo_memory["temporary_keyframes"]["pose_data"] += get_lietensor_memory(kf.pose_mu)
            if kf.pose_std is not None:
                hypo_memory["temporary_keyframes"]["pose_data"] += get_lietensor_memory(kf.pose_std)
            if kf.pose_weights is not None:
                hypo_memory["temporary_keyframes"]["pose_data"] += get_tensor_memory(kf.pose_weights)
    
    hypo_memory["temporary_keyframes"]["count"] = temp_kf_count
    hypo_memory["temporary_keyframes"]["total"] = (
        hypo_memory["temporary_keyframes"]["rgb_images"] + 
        hypo_memory["temporary_keyframes"]["depth_images"] + 
        hypo_memory["temporary_keyframes"]["pose_data"]
    )
    
    # Odometry edges
    for edge in system.hypothesis_manager.odom_edges.values():
        hypo_memory["edges"]["odom_edges_memory"] += get_lietensor_memory(edge.mean)
        hypo_memory["edges"]["odom_edges_memory"] += get_lietensor_memory(edge.std)
        hypo_memory["edges"]["odom_edges_memory"] += get_tensor_memory(edge.information)
    
    # Visual edges
    visual_edge_count = 0
    for hypo in system.hypothesis_manager.hypotheses.values():
        for edge_list in hypo.visual_edges.values():
            visual_edge_count += len(edge_list)
            for edge in edge_list:
                hypo_memory["edges"]["visual_edges_memory"] += get_lietensor_memory(edge.mean)
                hypo_memory["edges"]["visual_edges_memory"] += get_lietensor_memory(edge.std)
    
    hypo_memory["edges"]["visual_edges_count"] = visual_edge_count
    hypo_memory["edges"]["total"] = (
        hypo_memory["edges"]["odom_edges_memory"] + 
        hypo_memory["edges"]["visual_edges_memory"]
    )
    
    # Current distribution
    if system.hypothesis_manager.dist is not None:
        mu, sigma, weights = system.hypothesis_manager.dist
        hypo_memory["current_distribution"] += get_lietensor_memory(mu)
        hypo_memory["current_distribution"] += get_lietensor_memory(sigma)
        hypo_memory["current_distribution"] += get_tensor_memory(weights)
    
    if hasattr(system.hypothesis_manager, 'dist_weight_ema'):
        hypo_memory["current_distribution"] += get_tensor_memory(
            system.hypothesis_manager.dist_weight_ema
        )
    
    hypo_memory["total"] = (
        hypo_memory["temporary_keyframes"]["total"] + 
        hypo_memory["edges"]["total"] + 
        hypo_memory["current_distribution"]
    )
    
    memory_stats["hypothesis_manager"] = hypo_memory
    
    # --- 3. Models Memory ---
    models_memory = {
        "total": 0,
        "pose_estimation": 0,
        "depth_prediction": 0,
    }
    
    # Pose estimation model
    if hasattr(system, 'pose_est'):
        if hasattr(system.pose_est, 'model') and system.pose_est.model is not None:
            for param in system.pose_est.model.parameters():
                models_memory["pose_estimation"] += get_tensor_memory(param)
        # Also check for other model components
        if hasattr(system.pose_est, 'kp_detector') and hasattr(system.pose_est.kp_detector, 'net'):
            for param in system.pose_est.kp_detector.net.parameters():
                models_memory["pose_estimation"] += get_tensor_memory(param)
        if hasattr(system.pose_est, 'kp_matcher') and hasattr(system.pose_est.kp_matcher, 'extractor'):
            if hasattr(system.pose_est.kp_matcher.extractor, 'parameters'):
                for param in system.pose_est.kp_matcher.extractor.parameters():
                    models_memory["pose_estimation"] += get_tensor_memory(param)
    
    # Depth prediction model
    if system.use_depth_pred and hasattr(system, 'depth_pred'):
        if hasattr(system.depth_pred, 'model'):
            for param in system.depth_pred.model.parameters():
                models_memory["depth_prediction"] += get_tensor_memory(param)
    
    models_memory["total"] = (
        models_memory["pose_estimation"] + 
        models_memory["depth_prediction"]
    )
    
    memory_stats["models"] = models_memory
    

    # --- 7. System-wide memory info ---
    process = psutil.Process()
    memory_info = process.memory_info()
    
    memory_stats["system_wide"] = {
        "rss_mb": bytes_to_mb(memory_info.rss),  # Resident Set Size
        "vms_mb": bytes_to_mb(memory_info.vms),  # Virtual Memory Size
    }
    
    # Add CUDA memory if available
    if torch.cuda.is_available():
        memory_stats["cuda"] = {
            "allocated_mb": torch.cuda.memory_allocated(system.device) / (1024 * 1024),
            "reserved_mb": torch.cuda.memory_reserved(system.device) / (1024 * 1024),
        }
    
    memory_stats["total"] = (
        db_memory["total"] + 
        hypo_memory["total"] + 
        models_memory["total"] + 
        memory_stats["system_wide"]["rss_mb"] + 
        memory_stats["system_wide"]["vms_mb"]
    )
    # Convert all memory values to MB
    if detailed:
        memory_stats_mb = convert_dict_to_mb(memory_stats)
    else:
        memory_stats_mb = {
            "database_mb": bytes_to_mb(db_memory["total"]),
            "hypothesis_manager_mb": bytes_to_mb(hypo_memory["total"]),
            "models_mb": bytes_to_mb(models_memory["total"]),
            "total_mb": bytes_to_mb(memory_stats["total"]),
            "system_wide": memory_stats["system_wide"],
        }
        if "cuda" in memory_stats:
            memory_stats_mb["cuda"] = memory_stats["cuda"]
    
    return memory_stats_mb
    
def convert_dict_to_mb(d):
    """Recursively convert byte values to MB in a dictionary."""
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result[key] = convert_dict_to_mb(value)
        elif isinstance(value, (int, float)) and key not in [
            "count", "hypotheses_count", "odom_edges_count", "visual_edges_count"
        ]:
            result[key] = value / (1024 * 1024) if key not in ["rss_mb", "vms_mb", "allocated_mb", "reserved_mb"] else value
        else:
            result[key] = value
    return result

def print_memory_usage(system, detailed: bool = True):
    """Print memory usage statistics in a formatted way.
    
    Args:
        detailed: If True, print detailed breakdown; if False, only print summary
    """
    memory_stats = get_memory_usage(system, detailed=detailed)
    
    logger.info("="*80)
    logger.info("SYSTEM MEMORY USAGE")
    logger.info("="*80)
    
    if detailed:
        # Database
        logger.info("\n--- DATABASE ---")
        db = memory_stats["database"]
        logger.info(f"  Keyframes ({db['keyframes']['count']} total):")
        logger.info(f"    RGB Images:    {db['keyframes']['rgb_images']:.2f} MB")
        logger.info(f"    Depth Images:  {db['keyframes']['depth_images']:.2f} MB")
        logger.info(f"    Pose Data:     {db['keyframes']['pose_data']:.2f} MB")
        logger.info(f"    Subtotal:      {db['keyframes']['total']:.2f} MB")
        logger.info(f"  Embeddings:      {db['embeddings']:.2f} MB")
        logger.info(f"  VPR Model:       {db['vpr_model']:.2f} MB")
        logger.info(f"  Total:           {db['total']:.2f} MB")
        
        # Hypothesis Manager
        logger.info("\n--- HYPOTHESIS MANAGER ---")
        hypo = memory_stats["hypothesis_manager"]
        logger.info(f"  Hypotheses:      {hypo['hypotheses_count']} active")
        logger.info(f"  Temporary KFs ({hypo['temporary_keyframes']['count']} total):")
        logger.info(f"    RGB Images:    {hypo['temporary_keyframes']['rgb_images']:.2f} MB")
        logger.info(f"    Depth Images:  {hypo['temporary_keyframes']['depth_images']:.2f} MB")
        logger.info(f"    Pose Data:     {hypo['temporary_keyframes']['pose_data']:.2f} MB")
        logger.info(f"    Subtotal:      {hypo['temporary_keyframes']['total']:.2f} MB")
        logger.info(f"  Edges:")
        logger.info(f"    Odometry ({hypo['edges']['odom_edges_count']}):  {hypo['edges']['odom_edges_memory']:.2f} MB")
        logger.info(f"    Visual ({hypo['edges']['visual_edges_count']}):    {hypo['edges']['visual_edges_memory']:.2f} MB")
        logger.info(f"    Subtotal:      {hypo['edges']['total']:.2f} MB")
        logger.info(f"  Current Dist:    {hypo['current_distribution']:.2f} MB")
        logger.info(f"  Total:           {hypo['total']:.2f} MB")
        
        # Models
        logger.info("\n--- MODELS ---")
        models = memory_stats["models"]
        logger.info(f"  Pose Estimation: {models['pose_estimation']:.2f} MB")
        logger.info(f"  Depth Pred:      {models['depth_prediction']:.2f} MB")
        logger.info(f"  Total:           {models['total']:.2f} MB")

    
    # Summary
    logger.info("\n--- SUMMARY ---")
    if detailed:
        logger.info(f"  Database:        {memory_stats['database']['total']:.2f} MB")
        logger.info(f"  Hypo Manager:    {memory_stats['hypothesis_manager']['total']:.2f} MB")
        logger.info(f"  Models:          {memory_stats['models']['total']:.2f} MB")
        logger.info(f"  Tracked Total:   {memory_stats['total']:.2f} MB")
    else:
        logger.info(f"  Database:        {memory_stats['database_mb']:.2f} MB")
        logger.info(f"  Hypo Manager:    {memory_stats['hypothesis_manager_mb']:.2f} MB")
        logger.info(f"  Models:          {memory_stats['models_mb']:.2f} MB")
        logger.info(f"  Tracked Total:   {memory_stats['total_mb']:.2f} MB")
    
    # System-wide
    logger.info("\n--- SYSTEM-WIDE ---")
    sys_wide = memory_stats["system_wide"]
    logger.info(f"  Process RSS:     {sys_wide['rss_mb']:.2f} MB")
    logger.info(f"  Process VMS:     {sys_wide['vms_mb']:.2f} MB")
    
    if "cuda" in memory_stats:
        cuda = memory_stats["cuda"]
        logger.info(f"  CUDA Allocated:  {cuda['allocated_mb']:.2f} MB")
        logger.info(f"  CUDA Reserved:   {cuda['reserved_mb']:.2f} MB")
    
    logger.info("="*80)