"""Quick script to rename a checkpoint folder in Modal volume."""
import modal

app = modal.App("rename-checkpoint")
vol = modal.Volume.from_name("lidar-llm")

@app.function(volumes={"/data": vol}, timeout=300)
def move_folder(old_path: str, new_path: str):
    import os
    import shutil
    
    old_full = f"/data/{old_path}"
    new_full = f"/data/{new_path}"
    
    print(f"Moving: {old_full} -> {new_full}")
    
    if not os.path.exists(old_full):
        print(f"ERROR: Source path does not exist: {old_full}")
        return False
    
    # Create parent directory if needed
    new_parent = os.path.dirname(new_full)
    if not os.path.exists(new_parent):
        os.makedirs(new_parent)
        print(f"Created directory: {new_parent}")
    
    if os.path.exists(new_full):
        print(f"ERROR: Destination already exists: {new_full}")
        return False
    
    # Use shutil.move for cross-filesystem compatibility
    shutil.move(old_full, new_full)
    
    # Commit the volume changes
    vol.commit()
    
    print(f"Successfully moved to: {new_full}")
    
    # List contents to verify
    print("\nContents of /data/:")
    for item in os.listdir("/data"):
        print(f"  {item}/")
    
    print(f"\nContents of {new_parent}/:")
    for item in os.listdir(new_parent):
        print(f"  {item}")
    
    return True

@app.local_entrypoint()
def main():
    old_path = "checkpoints/run_20251201_111658"
    new_path = "checkpoints_backup/run_20251201_overfitting_baseline"
    
    print(f"Moving checkpoint folder...")
    print(f"  From: {old_path}")
    print(f"  To:   {new_path}")
    
    result = move_folder.remote(old_path, new_path)
    
    if result:
        print("\n✅ Move completed successfully!")
    else:
        print("\n❌ Move failed!")
