import torch
import torch_directml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_directml():
    try:
        # Initialize DirectML device
        device = torch_directml.device()
        logger.info(f"Successfully initialized DirectML device: {device}")
        
        # Create a simple tensor and move it to the device
        x = torch.randn(3, 3)
        x = x.to(device)
        logger.info(f"Successfully moved tensor to DirectML device")
        
        # Perform a simple operation
        y = x @ x
        logger.info(f"Successfully performed matrix multiplication on DirectML device")
        
        # Move result back to CPU and print
        result = y.cpu()
        logger.info(f"Result shape: {result.shape}")
        logger.info(f"Result:\n{result}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing DirectML: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_directml()
    if success:
        logger.info("DirectML test completed successfully!")
    else:
        logger.error("DirectML test failed!") 