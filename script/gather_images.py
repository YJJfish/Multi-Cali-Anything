import argparse
import os
import shutil
from typing import List

def gather_images(input_path : str, output_path : str):
    try:
        shutil.rmtree(output_path)
    except:
        pass
    try:
        os.mkdir(output_path)
    except:
        pass
    # Get all camera names.
    camera_names : List[str] = []
    for camera_name in os.listdir(input_path):
        camera_names.append(camera_name)
    # Get all frame names.
    frame_names : List[str] =  []
    for image_name in os.listdir(os.path.join(input_path, camera_names[0])):
        frame_names.append(image_name[:-4])
    # Gather images.
    # For each frame.
    for frame_name in frame_names:
        # Create a folder.
        os.mkdir(os.path.join(output_path, "frame{}".format(frame_name)))
        # For each camera.
        for camera_name in camera_names:
            # Copy the image.
            src_image_name = os.path.join(input_path, camera_name, frame_name+".png")
            dst_image_name = os.path.join(output_path, "frame{}".format(frame_name), "camera{}_frame{}.png".format(camera_name, frame_name))
            shutil.copy(src_image_name, dst_image_name)
    # Done.
    return        
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="gather_images.py",
        description="""
                        Python script that gathers images from multiface dataset into several image folders that colmap can use.
                        The images of multiface dataset are grouped into many folders. Each folder corresponds to a camera (We use the folder name as the "camera name").
                        Within each folder, there are many images. Each folder corresponds to a frame (We use the image name as the "frame name").
                        We need group images of different camera names and of the same frame name into one folder.
                        The new folders will be named as 'frame[frame_id]'.
                        The new images will be named as 'camera[camera_id]_frame[frame_id].png' and placed into the 'frame[frame_id]' folder.
                    """,
        allow_abbrev=True
    )
    parser.add_argument("input", help="Path to multiface dataset.")
    parser.add_argument("output", help="Path to output dataset folder.")
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    
    gather_images(input_path, output_path)