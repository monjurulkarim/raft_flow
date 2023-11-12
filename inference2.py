import sys
sys.path.append('RAFT/core')
from utils import flow_viz
from raft import RAFT
import torch
import numpy as np
import cv2
from collections import OrderedDict
from argparse import ArgumentParser
import os



def frame_preprocess(frame, device):
    frame = torch.from_numpy(frame).permute(2, 0, 1).float()
    frame = frame.unsqueeze(0)
    frame = frame.to(device)
    return frame


def vizualize_flow(img, flo, save, counter):
    # permute the channels and change device is necessary
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    flo = cv2.cvtColor(flo, cv2.COLOR_RGB2BGR)

    # save or concatenate, show images
    if save:
        # cv2.imwrite(f"demo_frames/frame_{str(counter)}.jpg", img_flo)

        # write the flo frame to the output video
        global out
        out.write(flo)
    else:
        img_flo = np.concatenate([img, flo], axis=0)
        cv2.imshow("Optical Flow", img_flo / 255.0)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            return False

    return True


def get_cpu_model(model):
    new_model = OrderedDict()
    # get all layer's names from model
    for name in model:
        # create new name and update new model
        new_name = name[7:]
        new_model[new_name] = model[name]
    return new_model


def inference(args):

    # get the RAFT model
    model = RAFT(args)
    # load pretrained weights
    # pretrained_weights = torch.load(args.model, map_location=torch.device('cuda'))
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to('cuda:1')
    device = 'cuda:1'

    # if torch.cuda.is_available():
    #     device = "cuda"
    #     # parallel between available GPUs
    #     model = torch.nn.DataParallel(model)
    #     # load the pretrained weights into model
    #     model.load_state_dict(pretrained_weights)
    #     model.to(device)
    # else:
    #     device = "cpu"
    #     # change key names for CPU runtime
    #     pretrained_weights = get_cpu_model(pretrained_weights)
    #     # load the pretrained weights into model
    #     model.load_state_dict(pretrained_weights)

    # change model's mode to evaluation
    model.eval()

    video_dir = args.video
    files = os.listdir(video_dir)
    files = files[161:322]
    # print(files)
    vid_cnt = 0
    for vid in files:
        video_name = vid.split('.')[0] #video name without extension
        video_path = video_dir + vid #video name with directory
        print('================')
        print(vid_cnt, ':', video_path)
        print('================')
        vid_cnt +=1
        # get video name without extension
        # video_name = video_path[video_path.rfind('/') + 1:video_path.rfind('.')]

        # capture the video and get the first frame
        cap = cv2.VideoCapture(video_path)
        ret, frame_1 = cap.read()

        # determine if we want to save the flow video
        save = args.save
        folder_dir = 'hevi_train_flow/'
        if save:
            if not os.path.exists(folder_dir):
                os.mkdir(folder_dir)

            # prepare for output video
            global fps, fourcc, out
            fps = cap.get(cv2.CAP_PROP_FPS)  # get fps
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(folder_dir + video_name + '.mp4', fourcc, fps,
                                  (int(cap.get(3)), int(cap.get(4))))  # last two params: fpms, (width, height)

        # frame preprocessing
        frame_1 = frame_preprocess(frame_1, device)

        counter = 0
        with torch.no_grad():
            while True:
                # read the next frame
                ret, frame_2 = cap.read()
                if not ret:
                    break
                # preprocessing
                frame_2 = frame_preprocess(frame_2, device)
                # predict the flow
                flow_low, flow_up = model(frame_1, frame_2, iters=args.iters, test_mode=True)
                # transpose the flow output and convert it into numpy array
                ret = vizualize_flow(frame_1, flow_up, save, counter)
                if not ret:
                    break
                frame_1 = frame_2
                counter += 1

        cap.release()
        if save:
            out.release()


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", help="restore checkpoint")
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--video", type=str, default="./hevi_train_vid100/")
    parser.add_argument("--save", action="store_true", help="save demo frames")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )

    args = parser.parse_args()
    inference(args)


if __name__ == "__main__":
    main()
