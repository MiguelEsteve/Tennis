
import torch
import glob
from tqdm import tqdm
import os
import shutil
import cv2
import numpy as np
from configs.log_conf import getLogger
from configs.configs import FRAMES_PATH, FLOWS_PATH
from .utils import flow_to_image, crop, normalise
from .flownet2S import FlowNet2S



LOGGER = getLogger(__name__)
    
def process_two_images(model:FlowNet2S, files:list):
    """
    Process two images into one flow image
    Args:
        model: The model to use
        files: a list of 2 image paths
    Returns:
    """

    LOGGER.debug(f'procesing flownet2s for [{os.path.basename(files[0])}, {os.path.basename(files[1])}]')

    if len(files) != 2:
        return None
    imgs = list()
    if isinstance(files[0], str) and isinstance(files[1], str) \
            and os.path.exists(files[0]) and os.path.exists(files[1]):
            imgs.append(cv2.cvtColor(cv2.imread(files[0]), cv2.COLOR_BGR2RGB))
            imgs.append(cv2.cvtColor(cv2.imread(files[1]), cv2.COLOR_BGR2RGB))
    else:
        return None, None

    imgs = crop(imgs)
    imgs = np.array(imgs)
    imgs = np.moveaxis(imgs, -1, 1)
    imgs = normalise(imgs).astype(np.float32)

    imgs = torch.from_numpy(imgs)
    imgs = torch.unsqueeze(imgs, axis=0)    # add batch axis
    
    flow = model.predict(imgs)  # run the model

    flow = flow.numpy()
    flow = np.squeeze(flow)
    flow = np.transpose(flow, axes=(1, 2, 0))
    img = flow_to_image(flow)
    h, w = img.shape[0:2]
    new_h, new_w = int(h/4.0), int(w/4.0)
    img = cv2.resize(img, (new_w, new_h))

    LOGGER.debug('flownet ended')

    return img, flow


def infer_flow_and_save(input_images:list, model: FlowNet2S = None, output_dir:str=None):
        
    model = FlowNet2S.get_from_checkpoint() if model is None else model
    output_dir = TEST_IMAGES if output_dir is None else output_dir

    img, flow = process_two_images(model, input_images)
    dir, file = os.path.split(input_images[0])
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


    image1, image2 = input_images

    shutil.copy(image1, TEST_IMAGES + f'/{os.path.basename(image1)}')
    shutil.copy(image2, TEST_IMAGES + f'/{os.path.basename(image2)}')

    out_image_name = 'flow_' + os.path.basename(image1)[:-4] + '_' + os.path.basename(image2)[:-4] + '.jpg'
    
    cv2.imwrite(os.path.join(output_dir, out_image_name), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


    
def process_image_dir(model: FlowNet2S = None, input_dir:str = None, output_dir:str = None, debug:str = None):
    """
    Process a directory of images
    Args:
        model: The flownet model
        input_dir: The input image dir
        output_dir: The output image dir
    Returns: output path of last saved sample

    """
    model = FlowNet2S.get_from_checkpoint() if model == None else model
    input_dir = FRAMES_PATH if input_dir is None else input_dir
    output_dir = FLOWS_PATH if output_dir is None else output_dir

    for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
        files = glob.glob(input_dir + "/**/*" + ext, recursive=True)
        if len(files) > 0:
            break

    print(f'len files: {len(files)}')

    if not len(files) > 0:
        print("Couldn't find any files in {}".format(input_dir))
        return None

    files.sort()

    for i in tqdm(range(1, len(files)), desc='Calculating Flow'):
        # files_ = files[i-1: i+1]
        img, flow = process_two_images(model, files[i-1:i+1])
        dir, file = os.path.split(files[i])
        if int(file[:-4]) == 0:  # skip first frame of any video (assume numbered 0s)
            continue

        output_path = dir.replace(input_dir, output_dir)  # this keeps the recursive dir structure
        os.makedirs(output_path, exist_ok=True)
        if debug != None and i >= debug:
            break

        cv2.imwrite(os.path.join(output_path, file), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return output_path


def generate_flows(image_dir: str = None, flow_dir: str = None, debug:str=None):
    image_dir = FLOWS_PATH if image_dir is None else image_dir
    flow_dir = FLOWS_PATH if flow_dir is None else flow_dir

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FlowNet2S.get_from_checkpoint()
    model.to(device)
    process_image_dir(model, debug=debug)

    """
    ctx = mx.gpu(0)
    net = get_flownet(ctx=ctx)
    net.hybridize()
    process_imagedir(net, input_dir=image_dir, output_dir=flow_dir, ctx=ctx)
    """