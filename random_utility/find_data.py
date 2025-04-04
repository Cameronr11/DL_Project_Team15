import os
import glob
import sys


#Helper file to find the data files and directories
#It also allow us to run the script from any directory in one of the other files
def get_project_root():
    """Returns the absolute path to the project root directory"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def find_data_files():
    # Get absolute path to project root
    project_root = get_project_root()
    
    print(f"Project root: {project_root}")
    
    # List all directories in the project
    print("\n=== Top-level directories in project ===")
    for item in os.listdir(project_root):
        if os.path.isdir(os.path.join(project_root, item)):
            print(f"- {item}")
    
    # Look for processed_data directory
    processed_data_dir = os.path.join(project_root, 'processed_data')
    if os.path.exists(processed_data_dir):
        print(f"\nprocessed_data directory exists at: {processed_data_dir}")
        
        # List contents of processed_data
        print("\n=== Contents of processed_data ===")
        for item in os.listdir(processed_data_dir):
            item_path = os.path.join(processed_data_dir, item)
            if os.path.isdir(item_path):
                print(f"- {item} (directory)")
                # Count files in this directory
                files = os.listdir(item_path)
                print(f"  Contains {len(files)} files")
                if files:
                    # Show a few example filenames
                    print(f"  Examples: {', '.join(files[:5])}")
            else:
                print(f"- {item} (file)")
    else:
        print(f"\nprocessed_data directory NOT FOUND at: {processed_data_dir}")
        
        # Search for any .npy files in the project
        print("\nSearching for .npy files in the project...")
        npy_files = []
        for root, dirs, files in os.walk(project_root):
            for file in files:
                if file.endswith('.npy'):
                    npy_files.append(os.path.join(root, file))
        
        if npy_files:
            print(f"Found {len(npy_files)} .npy files.")
            print("First few examples:")
            for path in npy_files[:5]:
                print(f"- {path}")
        else:
            print("No .npy files found in the project.")
    
    # Check the data directory structure
    data_dir = os.path.join(project_root, 'data', 'MRNet-v1.0')
    if os.path.exists(data_dir):
        print(f"\ndata/MRNet-v1.0 directory exists at: {data_dir}")
        print("\n=== Contents of data/MRNet-v1.0 ===")
        for item in os.listdir(data_dir):
            print(f"- {item}")
    else:
        print(f"\ndata/MRNet-v1.0 directory NOT FOUND at: {data_dir}")

if __name__ == "__main__":
    find_data_files()