import uuid
from collections import namedtuple
from typing import List
import numpy as np
import cv2, os, argparse, audio
import subprocess
from tqdm import tqdm
import torch, face_detection
from models import Wav2Lip
import platform
import face_alignment


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


def main(face: str, audio_path: str, fa: face_alignment.FaceAlignment, model: Wav2Lip,
         fps: float = 25., resize_factor: int = 1, rotate: bool = False,
         wav2lip_batch_size: int = 128, crop: List = [0, -1, 0, -1], outfile: str = '',
         box: List = [-1, -1, -1, -1], static: bool = False, face_det_batch_size: int = 16, pads: List = [0, 10, 0, 0],
         nosmooth: bool = False, img_size: int = 288,
         device: str = "cuda", mel_step_size: int = 16):
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

    tmp_audio = None
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
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

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
    # 原全帧与推理全帧对
    fullFramePair = namedtuple("fullFramePair", ["org_full_frame", "pred_full_frame"])
    full_frame_pairs: List[fullFramePair] = []
    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                                                                    total=int(
                                                                        np.ceil(float(len(mel_chunks)) / batch_size)))):
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            org_full_frame = f.copy()
            f[y1:y2, x1:x2] = p
            full_frame_pairs.append(fullFramePair(org_full_frame=org_full_frame, pred_full_frame=f))

    # 推理的人脸遮罩点 抠图 ToDo refer to https://github.com/Rudrabha/Wav2Lip/issues/415
    pred_batch_landmarks = fa.get_landmarks_from_batch(
        torch.Tensor(
            np.stack([full_frame_pair.pred_full_frame for full_frame_pair in full_frame_pairs]).transpose(0, 3, 1, 2)))
    # 无声视频
    frame_h, frame_w = full_frames[0].shape[:-1]
    tmp_video = f"/dev/shm/{uuid.uuid4()}.avi"
    out = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
    for full_frame_pair, points_68 in zip(full_frame_pairs, pred_batch_landmarks):
        p = full_frame_pair.pred_full_frame  # 推理全帧
        frame_ndarray = full_frame_pair.org_full_frame  # 原全帧

        assert points_68.shape[0] >= 17
        face_points = points_68[:17]
        if points_68.shape[0] >= 25:
            face_points = np.append(face_points, [points_68[24], points_68[19]], axis=0)
        face_points = np.stack(face_points).astype(np.int32)
        # 1. 创建一个长方形遮罩
        mask = np.zeros(p.shape[:2], dtype=np.uint8)
        # 2. 使用fillPoly绘制人脸遮罩
        cv2.fillPoly(mask, [face_points], (255, 255, 255))
        # 反向遮罩
        reverse_mask = cv2.bitwise_not(mask)
        # 3. 使用遮罩提取人脸
        face_image = cv2.bitwise_and(p, p, mask=mask)
        # 提取人脸周围
        face_surrounding = cv2.bitwise_and(frame_ndarray, frame_ndarray, mask=reverse_mask)
        # 推理出的人脸贴回原帧
        joined_frame = cv2.add(face_image, face_surrounding)
        out.write(joined_frame)

    out.release()

    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_path, tmp_video, outfile)
    subprocess.call(command, shell=platform.system() != 'Windows')
    # 清理临时文件
    if tmp_audio:
        os.remove(tmp_audio)
    os.remove(tmp_video)


_fa = None


def get_fa(device: str = "cuda"):
    """获取FaceAlignment实例"""
    global _fa
    if _fa is None:
        print(f"load face_alignment model")
        _fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_HALF_D, device=device, face_detector='blazeface')
    return _fa


if __name__ == '__main__':
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

    args = parser.parse_args()
    # args.img_size = 96
    args.img_size = 288
    # output file
    if not args.outfile:
        args.outfile = os.path.join("results", os.path.basename(args.checkpoint_path) + ".mp4")

    if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        args.static = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} for inference.'.format(device))

    model = load_model(args.checkpoint_path, device)
    print("Model loaded")
    main(model=model, face=args.face, fps=args.fps, resize_factor=args.resize_factor, rotate=args.rotate,
         audio_path=args.audio,
         wav2lip_batch_size=args.wav2lip_batch_size, crop=args.crop, outfile=args.outfile,
         fa=get_fa(device), box=args.box, static=args.static, face_det_batch_size=args.face_det_batch_size,
         pads=args.pads, nosmooth=args.nosmooth, img_size=args.img_size,
         device=device, mel_step_size=16)
