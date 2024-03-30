from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
import glob
import argparse
import os
import random
from shutil import copyfile

from yaml import safe_load
import yaml


def tiler(imnames: List[str], out_image_path: Path, out_label_path: Path, slice_size: int, ext: str):
    for imname in imnames:
        im = Image.open(imname)
        imr = np.array(im, dtype=np.uint8)
        height = imr.shape[0]
        width = imr.shape[1]
        labname = imname.replace(ext, 'txt').replace("images", "labels")
        labels = pd.read_csv(labname, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])
        
        # we need to rescale coordinates from 0-1 to real image height and width
        labels[['x1', 'w']] = labels[['x1', 'w']] * width
        labels[['y1', 'h']] = labels[['y1', 'h']] * height
        
        boxes = []
        
        # convert bounding boxes to shapely polygons. We need to invert Y and find polygon vertices from center points
        for row in labels.iterrows():
            x1 = row[1]['x1'] - row[1]['w']/2
            y1 = (height - row[1]['y1']) - row[1]['h']/2
            x2 = row[1]['x1'] + row[1]['w']/2
            y2 = (height - row[1]['y1']) + row[1]['h']/2

            boxes.append((int(row[1]['class']), Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])))
        
        counter = 0
        # create tiles and find intersection with bounding boxes for each tile
        for i in range((height // slice_size)):
            for j in range((width // slice_size)):
                x1 = j*slice_size
                y1 = height - (i*slice_size)
                x2 = ((j+1)*slice_size) - 1
                y2 = (height - (i+1)*slice_size) + 1

                pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                imsaved = False
                slice_labels = []

                for box in boxes:
                    if pol.intersects(box[1]):
                        inter = pol.intersection(box[1])        
                        
                        if not imsaved:
                            sliced = imr[i*slice_size:(i+1)*slice_size, j*slice_size:(j+1)*slice_size]
                            sliced_im = Image.fromarray(sliced)
                            filename = imname.split('/')[-1]
                            slice_path = out_image_path / filename.replace(f".{ext}", f'_{i}_{j}.{ext}')                            
                            slice_labels_path = out_label_path / filename.replace(f".{ext}", f'_{i}_{j}.txt')                            
                            sliced_im.save(slice_path)
                            imsaved = True                    
                    
                        # get smallest rectangular polygon (with sides parallel to the coordinate axes) that contains the intersection
                        new_box = inter.envelope 
                        
                        # get central point for the new bounding box 
                        centre = new_box.centroid
                        
                        # get coordinates of polygon vertices
                        x, y = new_box.exterior.coords.xy
                        
                        # get bounding box width and height normalized to slice size
                        new_width = (max(x) - min(x)) / slice_size
                        new_height = (max(y) - min(y)) / slice_size
                        
                        # we have to normalize central x and invert y for yolo format
                        new_x = (centre.coords.xy[0][0] - x1) / slice_size
                        new_y = (y1 - centre.coords.xy[1][0]) / slice_size
                        
                        counter += 1

                        slice_labels.append([box[0], new_x, new_y, new_width, new_height])
                
                if len(slice_labels) > 0:
                    slice_df = pd.DataFrame(slice_labels, columns=['class', 'x1', 'y1', 'w', 'h'])
                    slice_df.to_csv(slice_labels_path, sep=' ', index=False, header=False, float_format='%.6f')
                
                if not imsaved:
                    sliced = imr[i*slice_size:(i+1)*slice_size, j*slice_size:(j+1)*slice_size]
                    sliced_im = Image.fromarray(sliced)
                    filename = imname.split('/')[-1]
                    slice_path = out_image_path / filename.replace(f".{ext}", f'_{i}_{j}.{ext}')                
                    slice_labels_path = out_label_path / filename.replace(f".{ext}", f'_{i}_{j}.txt')                            

                    sliced_im.save(slice_path)
                    imsaved = True

if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-source", default="./yolosample/ts/", help = "path to yaml file that describes the dataset")
    parser.add_argument("-target", default="./yolosliced/ts/", help = "Target folder for a new sliced dataset")
    parser.add_argument("-size", type=int, default=640, help = "Size of a tile. Dafault: 416")
    parser.add_argument("-ext", type=str, default="jpg", help = "Picture file format")

    args = parser.parse_args()

    source_path = Path(args.source)
    out_path = Path(args.target)

    print(source_path)

    with open(source_path) as stream:
        try:
            yaml_file = safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    train_imgs_path = source_path.parent / yaml_file["path"] / yaml_file["train"] / '*'
    val_imgs_path = source_path.parent / yaml_file["path"] / yaml_file["val"] / '*'
    test_imgs_path = source_path.parent / yaml_file["path"] / yaml_file["test"] / '*'

    train_imgs_path_out = out_path / yaml_file["train"] 
    val_imgs_path_out = out_path / yaml_file["val"]
    test_imgs_path_out = out_path / yaml_file["test"]
    train_labels_path_out = out_path / "labels/train"
    val_labels_path_out = out_path / "labels/val"

    train_imgs = glob.glob(str(train_imgs_path))
    val_imgs = glob.glob(str(val_imgs_path))
    test_imgs = glob.glob(str(test_imgs_path))

    train_labels = str(train_imgs_path).replace("images", "labels")
    val_labels = str(val_imgs_path).replace("images", "labels")

    if os.path.exists(args.target):
        raise Exception("Target folder should not exist yet")
    
    os.makedirs(train_imgs_path_out)
    os.makedirs(val_imgs_path_out)
    os.makedirs(test_imgs_path_out)
    os.makedirs(train_labels_path_out)
    os.makedirs(val_labels_path_out)   

    with open(str(out_path / args.target) + ".yaml", "w" ) as out_yaml:
        out_config = yaml_file.copy()
        out_config["path"] = ""
        yaml.dump(yaml_file, out_yaml, default_flow_style=False)

    # # classes.names should be located one level higher than images   
    # # this file is not changing, so we will just copy it to a target folder 
    # upfolder = os.path.join(args.source, '..' )
    # target_upfolder = os.path.join(args.target, '..' )
    # if not os.path.exists(os.path.join(upfolder, 'classes.names')):
    #     print('classes.names not found. It should be located one level higher than images')
    # else:
    #     copyfile(os.path.join(upfolder, 'classes.names'), os.path.join(target_upfolder, 'classes.names'))
    
    # if args.falsefolder:
    #     if not os.path.exists(args.falsefolder):
    #         os.makedirs(args.falsefolder)
    #     elif len(os.listdir(args.falsefolder)) > 0:
    #         raise Exception("Folder for tiles without boxes should be empty")

    tiler(train_imgs, train_imgs_path_out, train_labels_path_out, args.size, args.ext)
    tiler(val_imgs, val_imgs_path_out, val_labels_path_out, args.size, args.ext)

