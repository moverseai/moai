import cv2
import torch
import numpy
import io

#NOTE: extract these to common loading funcs

def load_image(filename: str, data_type=torch.float32):
    color_img = numpy.array(cv2.imread(filename, cv2.IMREAD_ANYCOLOR))
    color_img =cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    h, w, c = color_img.shape
    color_data = color_img.astype(numpy.float32).transpose(2, 0, 1)
    return torch.from_numpy(
        color_data.reshape(1, c, h, w)        
    ).type(data_type) / 255.0

def load_depth(filename: str, data_type=torch.float32, scale=0.001):
    depth_img = numpy.array(cv2.imread(filename, cv2.IMREAD_ANYDEPTH))
    h, w = depth_img.shape
    depth_data = depth_img.astype(numpy.float32) * scale
    return torch.from_numpy(
        depth_data.reshape(1, 1, h, w)        
    ).type(data_type)

def crop_depth(filename: str, data_type=torch.float32, scale=0.001):
    depth_img = numpy.array(cv2.imread(filename, cv2.IMREAD_ANYDEPTH))
    center_cropped_depth_img = depth_img[60:420, 0:640]
    h, w = center_cropped_depth_img.shape
    depth_data = center_cropped_depth_img.astype(numpy.float32) * scale
    return torch.from_numpy(
        depth_data.reshape(1, 1, h, w)        
    ).type(data_type)

def readpgm(name: str): #NOTE: for ascii PGM
    with open(name) as f:
        lines = f.readlines()
    # Ignores commented lines
    for l in list(lines):
        if l[0] == '#':
            lines.remove(l)
    # Makes sure it is ASCII format (P2)
    assert lines[0].strip() == 'P2' 
    # Converts data to a list of integers
    data = []
    for line in lines[1:]:
        data.extend([int(c) for c in line.split()])

    buffer = numpy.array(data[3:])
    return buffer.reshape(data[1], data[0])


def load_depth_pgm(filename: str, data_type=torch.float32, scale = 1):
    depth_img = readpgm(filename)
    depth_img = depth_img.astype(numpy.float32) * scale
    h, w = depth_img.shape
    depth_data = depth_img.astype(numpy.float32)
    return torch.from_numpy(
        depth_data.reshape(1, 1, h, w)        
    ).type(data_type)
    
# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = numpy.frombuffer(buf.getvalue(), dtype=numpy.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(
        img.reshape(1, img.shape[2], img.shape[0], img.shape[1])        
    ).type(torch.float32) / 255.0