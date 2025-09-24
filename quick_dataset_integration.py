#!/usr/bin/env python3
"""
Quick Dataset Integration Script
Simple script to quickly add a new image folder to your cattle vs buffalo classification system
"""

import os
import shutil
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_integrate_dataset(image_folder, animal_type, breed_name=None):
    """
    Quickly integrate a new image folder into the existing system
    
    Args:
        image_folder (str): Path to folder containing images
        animal_type (str): "cattle" or "buffalo"
        breed_name (str): Specific breed name (optional)
    """
    
    print(f"🚀 Quick Dataset Integration")
    print(f"📁 Image folder: {image_folder}")
    print(f"🐄 Animal type: {animal_type}")
    print(f"🏷️  Breed: {breed_name or 'Mixed'}")
    print("-" * 50)
    
    # Validate inputs
    source_path = Path(image_folder)
    if not source_path.exists():
        print(f"❌ Error: Folder '{image_folder}' does not exist")
        return False
    
    if animal_type not in ['cattle', 'buffalo']:
        print(f"❌ Error: Animal type must be 'cattle' or 'buffalo'")
        return False
    
    # Create directories
    project_root = Path(".")
    new_dataset_dir = project_root / "new_integrated_dataset"
    new_dataset_dir.mkdir(exist_ok=True)
    
    # Process images
    processed_count = 0
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    print(f"🔄 Processing images...")
    
    for img_file in source_path.rglob('*'):
        if img_file.suffix.lower() in supported_formats:
            try:
                # Load image
                img = cv2.imread(str(img_file))
                if img is not None:
                    # Resize to standard size
                    img_resized = cv2.resize(img, (224, 224))
                    
                    # Create new filename
                    new_filename = f"{animal_type}_{breed_name or 'mixed'}_{processed_count:04d}.jpg"
                    new_path = new_dataset_dir / new_filename
                    
                    # Save processed image
                    cv2.imwrite(str(new_path), img_resized)
                    processed_count += 1
                    
                    if processed_count % 10 == 0:
                        print(f"   Processed {processed_count} images...")
                        
            except Exception as e:
                print(f"⚠️  Warning: Failed to process {img_file.name}: {e}")
    
    print(f"✅ Successfully processed {processed_count} images")
    
    # Create metadata
    metadata = {
        'dataset_name': f"{animal_type}_{breed_name or 'mixed'}_dataset",
        'animal_type': animal_type,
        'breed': breed_name,
        'total_images': processed_count,
        'processed_date': datetime.now().isoformat(),
        'source_folder': str(image_folder),
        'status': 'ready_for_integration'
    }
    
    # Save metadata
    metadata_path = new_dataset_dir / "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Metadata saved: {metadata_path}")
    
    # Update existing metadata if available
    existing_metadata_path = project_root / "cattle_buffalo_model_metadata.json"
    if existing_metadata_path.exists():
        with open(existing_metadata_path, 'r') as f:
            existing_metadata = json.load(f)
        
        # Add new breed if specified
        if breed_name and breed_name != 'Unknown':
            if animal_type == 'cattle':
                if breed_name not in existing_metadata.get('cattle_breeds', []):
                    existing_metadata.setdefault('cattle_breeds', []).append(breed_name)
            elif animal_type == 'buffalo':
                if breed_name not in existing_metadata.get('buffalo_breeds', []):
                    existing_metadata.setdefault('buffalo_breeds', []).append(breed_name)
        
        # Update total count
        existing_metadata['total_images'] = existing_metadata.get('total_images', 0) + processed_count
        existing_metadata['last_updated'] = datetime.now().isoformat()
        
        # Save updated metadata
        with open(existing_metadata_path, 'w') as f:
            json.dump(existing_metadata, f, indent=2)
        
        print(f"📊 Updated existing metadata")
    
    # Create integration instructions
    instructions = f"""
🎉 Dataset Integration Complete!

📁 New dataset location: {new_dataset_dir}
📊 Total images processed: {processed_count}
🐄 Animal type: {animal_type}
🏷️  Breed: {breed_name or 'Mixed'}

Next steps:
1. Run the model retraining script to include this dataset
2. Test the updated system with new images
3. Update the breed database if needed

To retrain the model with this new dataset, run:
python retrain_with_new_dataset.py
"""
    
    print(instructions)
    
    # Save instructions
    instructions_path = new_dataset_dir / "integration_instructions.txt"
    with open(instructions_path, 'w') as f:
        f.write(instructions)
    
    return True

def main():
    """Main function for quick integration"""
    print("🚀 Quick Dataset Integration Tool")
    print("=" * 40)
    print()
    
    # Get user input
    image_folder = input("📁 Enter path to your image folder: ").strip()
    
    print("\n🐄 Animal type:")
    print("1. Cattle")
    print("2. Buffalo")
    animal_choice = input("Choose (1 or 2): ").strip()
    animal_type = "cattle" if animal_choice == "1" else "buffalo"
    
    breed_name = input("🏷️  Enter specific breed name (or press Enter for mixed): ").strip()
    if not breed_name:
        breed_name = None
    
    print(f"\n🔄 Starting integration...")
    print(f"   Image folder: {image_folder}")
    print(f"   Animal type: {animal_type}")
    print(f"   Breed: {breed_name or 'Mixed'}")
    print()
    
    success = quick_integrate_dataset(image_folder, animal_type, breed_name)
    
    if success:
        print("\n🎉 Integration completed successfully!")
    else:
        print("\n❌ Integration failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
