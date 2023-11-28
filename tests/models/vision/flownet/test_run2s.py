from glob import glob
import numpy as np
import os
from configs.configs import FRAMES_PATH, DATASETS_PATH, TEST_IMAGES
from models.vision.flownet.run2s import process_two_images, process_image_dir, generate_flows, infer_flow_and_save
from models.vision.flownet.flownet2S import FlowNet2S

def test_process_two_images():
    for ext in ['.png', 'jpg', '.jpeg']:
        files = glob(FRAMES_PATH + '/**/*' + ext, recursive=True)
        if len(files):
            break
    idx = np.random.choice(range(len(files)), 1)[0]
    images = files[idx-1:idx+1]
    model = FlowNet2S.get_from_checkpoint()
    process_two_images(model, images)

def test_infer_flow_and_save():
    image1 = os.path.join(DATASETS_PATH, 'frames/V006.mp4/0000011000/0000011050.jpg')
    image2 = os.path.join(DATASETS_PATH, 'frames/V006.mp4/0000011000/0000011056.jpg')
    image_list = [image1, image2]

    infer_flow_and_save(input_images=image_list)

def test_process_image_dir():
    process_image_dir(debug=2)

def test_generate_flows():
    generate_flows()

if __name__ == '__main__':
    # test_process_two_images()
    test_infer_flow_and_save()
    # test_process_image_dir()
    # test_generate_flows()


