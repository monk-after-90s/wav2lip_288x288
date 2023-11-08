import uuid
from typing import List
import numpy as np
import cv2, os, argparse, audio
import subprocess
from tqdm import tqdm
import torch, face_detection
from models import Wav2Lip
import platform
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images, face_det_batch_size, pads, nosmooth, device):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                            flip_input=False, device=device)

    batch_size = face_det_batch_size

    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image)  # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results


def datagen(frames, mels, box, static, face_det_batch_size, pads, nosmooth, img_size, wav2lip_batch_size, device):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if box[0] == -1:
        if not static:
            face_det_results = face_detect(frames,
                                           face_det_batch_size=face_det_batch_size,
                                           pads=pads,
                                           nosmooth=nosmooth,
                                           device=device)  # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([frames[0]],
                                           face_det_batch_size=face_det_batch_size,
                                           pads=pads,
                                           nosmooth=nosmooth,
                                           device=device)
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
    # 无限拼接
    frames = np.concatenate((frames, np.flip(frames, axis=0)), axis=0)
    face_det_results = face_det_results + face_det_results[::-1]

    for i, m in enumerate(mels):
        idx = 0 if static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (img_size, img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, img_size // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch


def _load(checkpoint_path, device):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path, device):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path, device)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


def face_mask_from_image(image, face_landmarks_detector):
    """
    Calculate face mask from image. This is done by

    Args:
        image: numpy array of an image
        face_landmarks_detector: mediapipa face landmarks detector
    Returns:
        A uint8 numpy array with the same height and width of the input image, containing a binary mask of the face in the image
    """
    # initialize mask
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # detect face landmarks
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    detection = face_landmarks_detector.detect(mp_image)

    if len(detection.face_landmarks) == 0:
        # no face detected - set mask to all of the image
        mask[:] = 1
        return mask

    # extract landmarks coordinates
    face_coords = np.array([[lm.x * image.shape[1], lm.y * image.shape[0]] for lm in detection.face_landmarks[0]])

    # calculate convex hull from face coordinates
    convex_hull = cv2.convexHull(face_coords.astype(np.float32))

    # apply convex hull to mask
    return cv2.fillPoly(mask, pts=[convex_hull.squeeze().astype(np.int32)], color=1)


def main(face: str, audio_path: str, model: Wav2Lip,
         fps: float = 25., resize_factor: int = 1, rotate: bool = False,
         wav2lip_batch_size: int = 128, crop: List = [0, -1, 0, -1], outfile: str = '',
         box: List = [-1, -1, -1, -1], static: bool = False, face_det_batch_size: int = 16, pads: List = [0, 10, 0, 0],
         nosmooth: bool = False, img_size: int = 288,
         device: str = "cuda", mel_step_size: int = 16,
         face_landmarks_detector=None):
    tmp_video = ''
    tmp_audio = ''
    try:
        if not os.path.isfile(face):
            raise ValueError('--face argument must be a valid path to video/image file')

        elif face.split('.')[1] in ['jpg', 'png', 'jpeg']:
            full_frames = [cv2.imread(face)]
        else:
            video_stream = cv2.VideoCapture(face)
            fps = video_stream.get(cv2.CAP_PROP_FPS)

            print('Reading video frames...')

            full_frames = []
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                if resize_factor > 1:
                    frame = cv2.resize(frame, (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor))

                if rotate:
                    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

                y1, y2, x1, x2 = crop
                if x2 == -1: x2 = frame.shape[1]
                if y2 == -1: y2 = frame.shape[0]

                frame = frame[y1:y2, x1:x2]

                full_frames.append(frame)

        print("Number of frames available for inference: " + str(len(full_frames)))

        if not audio_path.endswith('.wav'):
            print('Extracting raw audio...')
            tmp_audio = f"/dev/shm/{uuid.uuid4()}.wav"
            command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_path, tmp_audio)

            subprocess.call(command, shell=True)
            audio_path = tmp_audio

        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)
        print(mel.shape)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError(
                'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_chunks = []
        mel_idx_multiplier = 80. / fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
            i += 1

        print("Length of mel chunks: {}".format(len(mel_chunks)))

        full_frames = full_frames[:len(mel_chunks)]

        batch_size = wav2lip_batch_size
        gen = datagen(full_frames, mel_chunks, box, static, face_det_batch_size,
                      pads, nosmooth, img_size, wav2lip_batch_size, device)

        # 无声视频
        frame_h, frame_w = full_frames[0].shape[:-1]
        tmp_video = f"/dev/shm/{uuid.uuid4()}.avi"
        out = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                                                                        total=int(
                                                                            np.ceil(
                                                                                float(len(mel_chunks)) / batch_size)))):
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

            with torch.no_grad():
                pred = model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                if face_landmarks_detector:
                    mask = face_mask_from_image(p, face_landmarks_detector)
                    f[y1:y2, x1:x2] = f[y1:y2, x1:x2] * (1 - mask[..., None]) + p * mask[..., None]
                else:
                    f[y1:y2, x1:x2] = p
                out.write(f)
        out.release()

        command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_path, tmp_video, outfile)
        subprocess.call(command, shell=platform.system() != 'Windows')
    finally:
        # 清理临时文件
        if os.path.exists(tmp_audio):
            os.remove(tmp_audio)
        if os.path.exists(tmp_video):
            os.remove(tmp_video)
        # 清理显存
        torch.cuda.empty_cache()


_fa = None

if __name__ == '__main__':
    # 命令行参数
    parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

    parser.add_argument('--checkpoint_path', type=str,
                        help='Name of saved checkpoint to load weights from',
                        required=True)

    parser.add_argument('--face', type=str,
                        help='Filepath of video/image that contains faces to use',
                        required=True)
    parser.add_argument('--audio', type=str,
                        help='Filepath of video/audio file to use as raw audio source',
                        required=True)
    parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.',
                        default='')

    parser.add_argument('--static', type=bool,
                        help='If True, then use only first video frame for inference', default=False)
    parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)',
                        default=25., required=False)

    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                        help='Padding (top, bottom, left, right). Please adjust to include chin at least')

    parser.add_argument('--face_det_batch_size', type=int,
                        help='Batch size for face detection', default=16)
    parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

    parser.add_argument('--resize_factor', default=1, type=int,
                        help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                        help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
                             'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

    parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                        help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                             'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

    parser.add_argument('--rotate', default=False, action='store_true',
                        help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                             'Use if you get a flipped result, despite feeding a normal looking video')

    parser.add_argument('--nosmooth', default=False, action='store_true',
                        help='Prevent smoothing face detections over a short temporal window')
    parser.add_argument('--face_landmarks_detector_path',
                        default='weights/face_landmarker_v2_with_blendshapes.task',
                        type=str,
                        help='Path to face landmarks detector. Pass empty string to ignore face landmarks detection.')
    args = parser.parse_args()
    # args.img_size = 96
    args.img_size = 288
    # output file
    if not args.outfile:
        args.outfile = os.path.join("results", os.path.basename(args.checkpoint_path) + ".mp4")

    if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        args.static = True
    # 推理设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} for inference.'.format(device))
    # 推理模型加载
    model = load_model(args.checkpoint_path, device)
    print("Model loaded")
    # 人脸遮罩检测器
    # Create an FaceLandmarker object.
    face_landmarks_detector = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=args.face_landmarks_detector_path),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1)
    ) if args.face_landmarks_detector_path else None
    try:
        # 推理主过程
        main(model=model, face=args.face, fps=args.fps, resize_factor=args.resize_factor, rotate=args.rotate,
             audio_path=args.audio, wav2lip_batch_size=args.wav2lip_batch_size, crop=args.crop, outfile=args.outfile,
             box=args.box, static=args.static, face_det_batch_size=args.face_det_batch_size,
             pads=args.pads, nosmooth=args.nosmooth, img_size=args.img_size,
             device=device, mel_step_size=16, face_landmarks_detector=face_landmarks_detector)
    finally:
        if hasattr(face_landmarks_detector, "close"):
            face_landmarks_detector.close()
