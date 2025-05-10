import torch
import logging
import os
import psutil
import gc

logger = logging.getLogger(__name__)

def get_available_memory():
    """Get available GPU memory in bytes."""
    try:
        # For DirectML, we'll use system memory as a proxy
        # since direct GPU memory query isn't available
        return psutil.virtual_memory().available
    except Exception as e:
        logger.warning(f"Could not get memory info: {e}")
        return None

def limit_memory_usage(percentage=70):
    """Limit memory usage to a percentage of available memory."""
    available = get_available_memory()
    if available:
        target_memory = int(available * (percentage / 100))
        # Set environment variable to limit DirectML memory usage
        os.environ['DIRECTML_MEMORY_LIMIT'] = str(target_memory)
        logger.info(f"Limited DirectML memory usage to {percentage}% ({target_memory / (1024**3):.2f} GB)")

def clear_memory():
    """Clear unused memory."""
    gc.collect()
    if torch.is_tensor:
        torch.cuda.empty_cache()  # This won't affect DirectML but keep for compatibility

def get_device():
    """Get the appropriate device for training."""
    try:
        # Try to use DirectML for AMD GPU
        import torch_directml
        dml = torch_directml.device()
        # Limit memory usage to 60%
        limit_memory_usage(65)
        logger.info(f"Using device: DirectML (AMD GPU)")
        return dml
    except ImportError:
        logger.warning("DirectML not available, falling back to CPU")
        return torch.device("cpu")
    except Exception as e:
        logger.warning(f"Error initializing DirectML: {e}. Falling back to CPU")
        return torch.device("cpu")

def memory_status():
    """Get current memory status."""
    try:
        available = get_available_memory()
        if available:
            total = psutil.virtual_memory().total
            used = total - available
            return {
                'total_gb': total / (1024**3),
                'used_gb': used / (1024**3),
                'available_gb': available / (1024**3),
                'usage_percent': (used / total) * 100
            }
    except Exception as e:
        logger.warning(f"Could not get memory status: {e}")
        return None 