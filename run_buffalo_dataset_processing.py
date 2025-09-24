#!/usr/bin/env python3
"""
Simple script to run Buffalo Breed Set processing
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("🐃 Buffalo Breed Set Dataset Processing")
    print("=" * 50)
    print()
    
    # Check if Buffalo Breed Set folder exists
    buffalo_folder = Path("Buffalo Breed Set")
    if not buffalo_folder.exists():
        print("❌ Error: 'Buffalo Breed Set' folder not found!")
        print("Please make sure the folder exists in the current directory.")
        return
    
    print(f"✅ Found 'Buffalo Breed Set' folder")
    print(f"📁 Location: {buffalo_folder.absolute()}")
    
    # Count images
    total_images = 0
    breed_count = 0
    for breed_folder in buffalo_folder.iterdir():
        if breed_folder.is_dir():
            breed_count += 1
            images_in_breed = len([f for f in breed_folder.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}])
            total_images += images_in_breed
            print(f"   🐃 {breed_folder.name}: {images_in_breed} images")
    
    print(f"\n📊 Total: {breed_count} breeds, {total_images} images")
    print()
    
    # Ask for confirmation
    response = input("Do you want to process this dataset? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ Processing cancelled")
        return
    
    print("\n🔄 Starting processing...")
    print("This may take a few minutes depending on the number of images.")
    print()
    
    try:
        # Run the processing script
        result = subprocess.run([sys.executable, "process_buffalo_breed_set.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Processing completed successfully!")
            print("\n📋 Output:")
            print(result.stdout)
        else:
            print("❌ Processing failed!")
            print("\n📋 Error output:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error running processing script: {e}")

if __name__ == "__main__":
    main()
