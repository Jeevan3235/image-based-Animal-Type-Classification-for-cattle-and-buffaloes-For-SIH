#!/usr/bin/env python3
"""
Quick Buffalo Breed Set Processing
Fast processing of the Buffalo Breed Set dataset
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

def quick_process_buffalo_dataset():
    """Quickly process the Buffalo Breed Set dataset"""
    print("🐃 Quick Buffalo Breed Set Processing")
    print("=" * 50)
    
    # Setup paths
    project_root = Path(".")
    buffalo_breed_set_path = project_root / "Buffalo Breed Set"
    processed_dataset_path = project_root / "processed_buffalo_dataset"
    enhanced_dataset_path = project_root / "enhanced_final_dataset"
    
    # Create directories
    processed_dataset_path.mkdir(exist_ok=True)
    enhanced_dataset_path.mkdir(exist_ok=True)
    
    if not buffalo_breed_set_path.exists():
        print("❌ Error: 'Buffalo Breed Set' folder not found!")
        return False
    
    # Buffalo breed mapping
    buffalo_breeds = {
        'Banni': 'Banni',
        'Bhadawari': 'Bhadawari', 
        'Chilika': 'Chilika',
        'Godawari': 'Godawari',
        'Gojri': 'Gojri',
        'Jaffarabadi': 'Jaffarabadi',
        'Kalahandi': 'Kalahandi',
        'Manda': 'Manda',
        'Marathwada': 'Marathwada',
        'Mehsana': 'Mehsana',
        'MURRAH': 'Murrah',
        'NAGPURI': 'Nagpuri',
        'NILI RAVI': 'Nili Ravi',
        'PANDHARPURI': 'Pandharpuri',
        'PARALAKHEMUNDI': 'Paralakhemundi',
        'SAMBALPURI': 'Sambalpuri',
        'South Kanara': 'South Kanara',
        'Surti': 'Surti',
        'Swamp': 'Swamp Buffalo',
        'Tarai': 'Tarai',
        'Toda': 'Toda'
    }
    
    processed_data = []
    total_images = 0
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    print(f"📁 Processing from: {buffalo_breed_set_path}")
    print(f"📁 Output to: {processed_dataset_path}")
    print()
    
    # Process each breed folder
    for breed_folder in buffalo_breed_set_path.iterdir():
        if breed_folder.is_dir():
            breed_name = buffalo_breeds.get(breed_folder.name, breed_folder.name)
            print(f"🔄 Processing {breed_name}...")
            
            breed_images = 0
            for img_file in breed_folder.iterdir():
                if img_file.suffix.lower() in supported_formats:
                    try:
                        # Load image
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            # Resize to standard size
                            img_resized = cv2.resize(img, (224, 224))
                            
                            # Create processed filename
                            processed_filename = f"buffalo_{breed_name}_{breed_images:04d}.jpg"
                            processed_path = processed_dataset_path / processed_filename
                            
                            # Save processed image
                            cv2.imwrite(str(processed_path), img_resized)
                            
                            processed_data.append({
                                'filename': processed_filename,
                                'path': str(processed_path),
                                'animal_type': 'buffalo',
                                'breed': breed_name,
                                'original_folder': breed_folder.name,
                                'original_file': img_file.name
                            })
                            
                            breed_images += 1
                            total_images += 1
                            
                    except Exception as e:
                        print(f"⚠️  Warning: Failed to process {img_file.name}: {e}")
            
            print(f"   ✅ {breed_name}: {breed_images} images processed")
    
    print(f"\n📊 Total processed: {total_images} images")
    
    # Save metadata
    metadata = {
        'dataset_name': 'Buffalo Breed Set',
        'animal_type': 'buffalo',
        'total_images': total_images,
        'total_breeds': len(buffalo_breeds),
        'processed_date': datetime.now().isoformat(),
        'breeds': list(buffalo_breeds.values()),
        'images': processed_data
    }
    
    metadata_path = processed_dataset_path / "buffalo_breed_set_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Metadata saved: {metadata_path}")
    
    # Copy to enhanced dataset
    print(f"\n🔄 Copying to enhanced dataset...")
    for img_file in processed_dataset_path.glob('*.jpg'):
        shutil.copy2(img_file, enhanced_dataset_path)
    
    print(f"✅ Enhanced dataset created: {enhanced_dataset_path}")
    
    # Update main metadata
    main_metadata_path = project_root / "cattle_buffalo_model_metadata.json"
    if main_metadata_path.exists():
        with open(main_metadata_path, 'r') as f:
            main_metadata = json.load(f)
    else:
        main_metadata = {
            "class_mapping": {},
            "breeds": [],
            "cattle_breeds": [],
            "buffalo_breeds": []
        }
    
    # Add buffalo breeds
    for breed in buffalo_breeds.values():
        if breed not in main_metadata.get('buffalo_breeds', []):
            main_metadata.setdefault('buffalo_breeds', []).append(breed)
    
    # Update counts
    main_metadata['total_images'] = main_metadata.get('total_images', 0) + total_images
    main_metadata['buffalo_images'] = total_images
    main_metadata['last_updated'] = datetime.now().isoformat()
    main_metadata['buffalo_breed_set_integrated'] = True
    
    # Save updated metadata
    with open(main_metadata_path, 'w') as f:
        json.dump(main_metadata, f, indent=2)
    
    print(f"📊 Updated main metadata: {main_metadata_path}")
    
    print(f"\n🎉 Buffalo Breed Set processing completed!")
    print(f"📁 Processed dataset: {processed_dataset_path}")
    print(f"📁 Enhanced dataset: {enhanced_dataset_path}")
    print(f"📊 Total images: {total_images}")
    print(f"🐃 Total breeds: {len(buffalo_breeds)}")
    
    return True

if __name__ == "__main__":
    success = quick_process_buffalo_dataset()
    if success:
        print("\n✅ Processing completed successfully!")
    else:
        print("\n❌ Processing failed!")

