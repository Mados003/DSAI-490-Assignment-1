import numpy as np
import sys
import os
from PIL import Image
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import process_path

def test_image_processing():    
    test_dir = "DummyClass"
    os.makedirs(test_dir, exist_ok=True)
    test_file_path = os.path.join(test_dir, "test_image.jpeg")
    
    dummy_image_data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    Image.fromarray(dummy_image_data, mode='L').save(test_file_path, format='JPEG')
    
    try:
        class_names = ["DummyClass", "OtherClass"]
        processed_img, label = process_path(test_file_path, class_names, img_size=(64, 64))
        
        assert tuple(processed_img.shape) == (1, 64, 64), f"Resize failed. Expected (1, 64, 64), got {processed_img.shape}"
        assert torch.max(processed_img) <= 1.0, "Normalization failed. Max value > 1.0"
        assert torch.min(processed_img) >= 0.0, "Normalization failed. Min value < 0.0"
        assert label == 0, f"Label extraction failed. Expected 0, got {label}"
        
        print("Data processing logic works perfectly!")
        
    finally:
        os.remove(test_file_path)
        os.rmdir(test_dir)

if __name__ == "__main__":
    test_image_processing()
    print("All data pipeline tests passed successfully!")