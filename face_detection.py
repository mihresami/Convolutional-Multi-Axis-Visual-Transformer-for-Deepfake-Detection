from tqdm import tqdm
import numpy as np
import cv2
import os
import argparse
import dlib
from imutils import face_utils
from numba import jit,cuda
from PIL import Image

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def detect_video(input_path, output_path, video_path, sample_len=30):
    path = os.path.join(output_path, video_path)
    if not os.path.exists(path):
        os.makedirs(path)
    vidcap = cv2.VideoCapture(os.path.join(input_path, video_path))
    frames = []
    landmarks = []
    count = 0
    while True:
        success, frame = vidcap.read()
        if success:
            frames.append(frame)
            count += 1
            if count >= sample_len:
                break
        else:
            print('fail')
            break
    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = detector(gray, 0)
        if len(face) == 0:
            print(path + 'failed!')
            return
        face = face[0]
        width = face.bottom() - face.top()
        landmark = predictor(frame, face)
        landmark = face_utils.shape_to_np(landmark)
        landmark = (landmark - np.array([face.left(), face.top()])) / width - 0.5
        landmarks.append(landmark.ravel().tolist())
        # 保存裁减图片
        
        img = frame[face.top():face.bottom(), face.left():face.right()]
        if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
            if width < 224:
                inter_para = cv2.INTER_CUBIC
            else:
                inter_para = cv2.INTER_AREA
            face_norm = cv2.resize(img, (224, 224), interpolation=inter_para)
            cv2.imwrite(os.path.join(path, str(i) + '.png'), face_norm)
    # 将landmark信息写入文件
    landmarks = np.array(landmarks)
    np.savetxt(os.path.join(path, 'landmarks.txt'), landmarks, fmt='%1.5f')
    # with open(os.path.join(path, 'landmarks.txt'),'a+') as outfile:
    # 	for land in landmarks:
    #  		np.savetxt(outfile, land, fmt='%1.5f')
    #/mnt/alpha/FaceForensics++/original_sequences/processed
    #/mnt/alpha/FaceForensics++/manipulated_sequences/NeuralTextures
    vidcap.release()
    return


def detect_video_2(input_path, output_path, video_path):
    path = os.path.join(output_path, video_path)
    # path = output_path
    if not os.path.exists(path):
        os.makedirs(path)
    for i, frame in enumerate(os.listdir(os.path.join(input_path,video_path))):
        # print(frame)
        frame = cv2.imread(os.path.join(input_path, video_path,frame))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = detector(gray, 0)
        if len(face) == 0:
            print((os.path.join(input_path, video_path)) + 'Failed no face detected!')
            continue
        else:
            face = face[0]
        # else len(face) > 0:
        #     face = face[0]
        # else:
        #     print("No face detected")
        #     continue
        width = face.bottom() - face.top()
        landmark = predictor(frame, face)
        landmark = face_utils.shape_to_np(landmark)
        landmark = (landmark - np.array([face.left(), face.top()])) / width - 0.5
        # landmarks.append(landmark.ravel().tolist())
        # 保存裁减图片
        img = frame[face.top():face.bottom(), face.left():face.right()]
        if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
            width, height, _ = img.shape  # Assuming BGR or RGB format for channels
            if width < 224:
                inter_para = cv2.INTER_CUBIC
                # print(f"Image width is less than 224, using CUBIC interpolation.")
            else:
                inter_para = cv2.INTER_AREA
                # print(f"Image width is 224 or greater, using AREA interpolation.")
            face_norm = cv2.resize(img, (224, 224), interpolation=inter_para)
            cv2.imwrite(os.path.join(path, str(i) + '.png'), face_norm)

        # img = frame[face.top():face.bottom(), face.left():face.right()]
        # if img is not None:
        #     if width < 224:
        #         inter_para = cv2.INTER_CUBIC
        #     else:
        #         inter_para = cv2.INTER_AREA
        #     face_norm = cv2.resize(img, (224, 224), interpolation=inter_para)
        #     cv2.imwrite(os.path.join(path, str(i) + '.png'), face_norm)

def data_process(args):
    input_path = args.input_path
    output_path = args.output_path
    for i,video in enumerate(tqdm(os.listdir(input_path))):
        if i<6000:
        # print(video)
            detect_video(input_path, output_path, video)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract faces and landmarks sequences from input videos.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input_path', type=str, default='./intput/',
                        help="Input videos path (folder)")
    parser.add_argument('-o', '--output_path', type=str, default='./output/',
                        help="Output datas path (folder)")
    args = parser.parse_args()
    data_process(args)
