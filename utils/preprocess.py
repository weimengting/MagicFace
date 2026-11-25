import os
import argparse
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from insightface.app import FaceAnalysis
from torchvision.utils import save_image

import data.datasets_faceswap as datasets_faceswap

pil2tensor = transforms.Compose([transforms.ToTensor(), transforms.Resize(512)])

pil2tensor = transforms.ToTensor()

app = FaceAnalysis(name='antelopev2', root=os.path.join('./',
                                                        'third_party_files'),
                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


def get_bbox(dets, crop_ratio):
    if crop_ratio > 0:
        bbox = dets[0:4]
        bbox_size = max(bbox[2] - bbox[0], bbox[2] - bbox[0])
        bbox_x = 0.5 * (bbox[2] + bbox[0])
        bbox_y = 0.5 * (bbox[3] + bbox[1])
        x1 = bbox_x - bbox_size * crop_ratio
        x2 = bbox_x + bbox_size * crop_ratio
        y1 = bbox_y - bbox_size * crop_ratio
        y2 = bbox_y + bbox_size * crop_ratio
        bbox_pts4 = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]], dtype=np.float32)
    else:
        # original box
        bbox = dets[0:4].reshape((2, 2))
        bbox_pts4 = datasets_faceswap.get_box_lm4p(bbox)
    return bbox_pts4



def crop_one_image(args):
    cur_img_sor_path = args.img_path
    im_pil_sor = Image.open(cur_img_sor_path).convert("RGB")
    face_info_sor = app.get(cv2.cvtColor(np.array(im_pil_sor), cv2.COLOR_RGB2BGR))
    assert len(face_info_sor) >= 1, 'The input image must contain a face！'
    if len(face_info_sor) > 1:
        print('The input image contain more than one face, we will only use the maximum face')
    face_info_sor = \
    sorted(face_info_sor, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * x['bbox'][3] - x['bbox'][1])[-1]
    dets_sor= face_info_sor['bbox']
    bbox_pst_sor = get_bbox(dets_sor, crop_ratio=0.75)

    warp_mat_crop_sor = datasets_faceswap.transformation_from_points(bbox_pst_sor,
                                                                     datasets_faceswap.mean_box_lm4p_512)
    im_crop512_sor = cv2.warpAffine(np.array(im_pil_sor), warp_mat_crop_sor, (512, 512), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)  # avoid black borders, which conflict with the black mask

    im_pil_sor = Image.fromarray(im_crop512_sor)
    im_pil_sor = pil2tensor(im_pil_sor)
    save_image(im_pil_sor, args.save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path",
        type=str,
        default='',
        required=False
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default='',
        required=False
    )
    args = parser.parse_args()
    crop_one_image(args)