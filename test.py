import os
import cv2
import numpy as np
import numexpr as ne
from modelhub.onnx import InsightFace2D106, InsightFaceSwap, YoloV5Face
# from xlib.torch import TorchDeviceInfo, get_cpu_device_info
from xlib.image.ImageProcessor import ImageProcessor
from xlib.face import ELandmarks2D, FLandmarks2D, FRect


def save_img(img, output_path):
    imgpath = os.path.join(output_path,'output.jpg')
    cv2.imwrite(imgpath, img)

"""
frame_image 原图
face_resolution cut时的resolution值 224
face_align_img  原图找出来的脸的图片 get_align_img(frame_image) 结果
face_align_mask_img  脸的蒙版 原图和生成图应该是一致的
face_align_lmrks_mask_img 生成原图脸图时的副产物
face_align_lmrks_mask_img = fsi.face_align_ulmrks.get_convexhull_mask( face_align_img.shape[:2], color=(255,), dtype=np.uint8)
face_swap_img 生成新脸的图 
face_swap_mask_img 生成新脸的遮罩 和 face_align_mask_img 一致

aligned_to_source_uni_mat 
aligned_to_source_uni_mat = image_to_align_uni_mat.invert()  image_to_align_uni_mat就是 face_mat
aligned_to_source_uni_mat = aligned_to_source_uni_mat.source_translated(-state.face_x_offset, -state.face_y_offset)
aligned_to_source_uni_mat = aligned_to_source_uni_mat.source_scaled_around_center(state.face_scale,state.face_scale)
aligned_to_source_uni_mat = aligned_to_source_uni_mat.to_exact_mat (face_width, face_height, frame_width, frame_height)

frame_width 
frame_height
frame_height, frame_width = merged_frame.shape[:2]


do_color_compression False
"""
def merge_face(frame_image, face_resolution, face_align_img, face_align_mask_img, face_align_lmrks_mask_img, face_swap_img, face_swap_mask_img, aligned_to_source_uni_mat, frame_width, frame_height, do_color_compression=False ):
    interpolation = ImageProcessor.Interpolation.LINEAR

    frame_image = ImageProcessor(frame_image).to_ufloat32().get_image('HWC')

    masks = []

    masks.append(ImageProcessor(face_align_mask_img).to_ufloat32().get_image('HW'))

    masks.append(ImageProcessor(face_swap_mask_img).to_ufloat32().get_image('HW'))

    # masks.append(ImageProcessor(face_align_lmrks_mask_img).to_ufloat32().get_image('HW'))

    masks_count = len(masks)
    if masks_count == 0:
        face_mask = np.ones(shape=(face_resolution, face_resolution), dtype=np.float32)
    else:
        face_mask = masks[0]
        for i in range(1, masks_count):
            face_mask *= masks[i]

    # Combine face mask
    face_mask_erode = 5
    face_mask_blur = 25
    face_mask = ImageProcessor(face_mask).erode_blur(face_mask_erode, face_mask_blur,
                                                     fade_to_border=True).get_image('HWC')
    frame_face_mask = ImageProcessor(face_mask).warp_affine(aligned_to_source_uni_mat, frame_width, frame_height).clip2(
        (1.0 / 255.0), 0.0, 1.0, 1.0).get_image('HWC')

    face_swap_ip = ImageProcessor(face_swap_img).to_ufloat32()

    face_swap_img = face_swap_ip.rct(like=face_align_img, mask=face_mask, like_mask=face_mask)

    frame_face_swap_img = face_swap_ip.warp_affine(aligned_to_source_uni_mat, frame_width, frame_height,
                                                   interpolation=interpolation).get_image('HWC')

    # Combine final frame
    face_opacity = 1
    opacity = np.float32(face_opacity)
    one_f = np.float32(1.0)
    if opacity == 1.0:
        out_merged_frame = ne.evaluate('frame_image*(one_f-frame_face_mask) + frame_face_swap_img*frame_face_mask')
    else:
        out_merged_frame = ne.evaluate(
            'frame_image*(one_f-frame_face_mask) + frame_image*frame_face_mask*(one_f-opacity) + frame_face_swap_img*frame_face_mask*opacity')

    return out_merged_frame


def get_face_ulmrks(frame_image):
    rects = face_detector.extract(frame_image, threshold=0.5)[0]
    _, H, W, _ = ImageProcessor(frame_image).get_dims()
    u_rects = [FRect.from_ltrb((l / W, t / H, r / W, b / H)) for l, t, r, b in rects]
    face_urect = FRect.sort_by_area_size(u_rects)[0]
    marker_coverage = 1.6
    face_image, face_uni_mat = face_urect.cut(frame_image, marker_coverage, 192)
    lmrks = face_marker.extract(face_image)[0]
    lmrks = lmrks[..., 0:2] / (192, 192)
    face_ulmrks = FLandmarks2D.create(ELandmarks2D.L106, lmrks)
    face_ulmrks = face_ulmrks.transform(face_uni_mat, invert=True)

    return face_ulmrks, face_uni_mat


def get_align_img(frame_image):
    face_ulmrks, face_uni_mat = get_face_ulmrks(frame_image)
    face_coverage = 2.2
    resolution = 224
    face_align_img, uni_mat = face_ulmrks.cut(frame_image, face_coverage,
                                  resolution,
                                  exclude_moving_parts=True,
                                  head_yaw=None,
                                  x_offset=0,
                                  y_offset=0 - 0.08,
                                  freeze_z_rotation=False)

    face_align_ulmrks = face_ulmrks.transform(uni_mat)
    face_align_lmrks_mask_img = face_align_ulmrks.get_convexhull_mask(face_align_img.shape[:2], color=(255,), dtype=np.uint8)

    return face_align_img, face_align_lmrks_mask_img, uni_mat

def read_img(img_path):
    frame = cv2.imread(img_path)
    # ip = ImageProcessor(frame)
    # frame = ip.get_image('HWC')
    return frame


def read_img_from_video(video_path, video_frame_index):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_index)
    ret, frame = cap.read()
    cap.release()
    if ret:
        ip = ImageProcessor(frame)
        frame = ip.get_image('HWC')
        return frame
    else:
        return None



def swap_face(source_img, target_img):
    face_ulmrks, _ = get_face_ulmrks(target_img)
    adjust_c = 1.55
    adjust_x = 0
    adjust_y = -0.15
    face_align_img, _ = face_ulmrks.cut(target_img, adjust_c,
                                        swap_model.get_face_vector_input_size(),
                                        x_offset=adjust_x,
                                        y_offset=adjust_y)

    # cv2.imshow('face_align_img', face_align_img)

    face_vector = swap_model.get_face_vector(face_align_img)

    if swap_model is not None and face_vector is not None:
        face_align_image = source_img
        if face_align_image is not None:
            _, H, W, _ = ImageProcessor(face_align_image).get_dims()
            anim_image = swap_model.generate(face_align_image, face_vector)
            anim_image = ImageProcessor(anim_image).resize((W, H)).get_image('HWC')

            white_mask = np.full_like(anim_image, 255, dtype=np.uint8)
            return anim_image, white_mask

    return None, None



def main():
    global swap_model
    global face_detector
    global face_marker
    available_devices = InsightFaceSwap.get_available_devices()
    cpu_device = available_devices[0]
    print(cpu_device)
    swap_model = InsightFaceSwap(cpu_device)
    face_detector = YoloV5Face(cpu_device)
    face_marker = InsightFace2D106(cpu_device)

    video_path = './test/1.mp4'
    pic_path = './test/2.png'
    output_path = './test/'
    video_frame_index = 170
    source_img = read_img_from_video(video_path, video_frame_index)
    align_img, face_align_lmrks_mask_img, face_mat = get_align_img(source_img)
    # cv2.imshow('source_img', source_img)
    # cv2.imshow('align_img', align_img)
    target_img = read_img(pic_path)
    # cv2.imshow('target_img', target_img)

    new_face_img, white_mask = swap_face(align_img, target_img)

    face_height, face_width = align_img.shape[:2]
    frame_height, frame_width = source_img.shape[:2]
    aligned_to_source_uni_mat = face_mat.invert()
    aligned_to_source_uni_mat = aligned_to_source_uni_mat.source_translated(-0, -0)
    aligned_to_source_uni_mat = aligned_to_source_uni_mat.source_scaled_around_center(1,1)
    aligned_to_source_uni_mat = aligned_to_source_uni_mat.to_exact_mat(face_width, face_height, frame_width,
                                                                       frame_height)

    out_merged_frame = merge_face(source_img,
               224,
               align_img, white_mask, face_align_lmrks_mask_img, new_face_img, white_mask, aligned_to_source_uni_mat, frame_width, frame_height, do_color_compression=False)


    if out_merged_frame is not None:
        # cv2.imshow('new_face_img', new_face_img)
        cv2.imshow('out_merged_frame', out_merged_frame)
        print(out_merged_frame)
        save_img(out_merged_frame*255, output_path)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()