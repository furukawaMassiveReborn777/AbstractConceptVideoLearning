# coding:utf-8
import torch, random
from PIL import Image, ImageOps

class VideoDataset(torch.utils.data.Dataset):

    def __init__(self, video_df, sample_num, img_ext, input_size, randomflip=False, transform=None):
        self.transform = transform
        self.video_df = video_df
        self.sample_num = sample_num
        self.img_ext = img_ext
        self.input_size = input_size
        self.randomflip = randomflip

    def __len__(self):
        return len(self.video_df)

    def __getitem__(self, idx):
        frame_dir = self.video_df["frame_dir"][idx]
        start_frame = self.video_df["start_frame"][idx]
        end_frame = self.video_df["end_frame"][idx]
        label = self.video_df["label"][idx]

        frame_total = end_frame - start_frame + 1
        if frame_total >= self.sample_num:
            # divide into approximate equal segment and select frame
            frame_per_seg = frame_total//self.sample_num
            seg_frames = [frame_per_seg] * self.sample_num
            for i in range(frame_total%self.sample_num):
                seg_frames[i] += 1
            start_idx = start_frame
            sample_indices = []
            for s in seg_frames:
                end_idx = start_idx + s - 1
                select_frame = random.randint(start_idx, end_idx)
                sample_indices.append(select_frame)
                start_idx += s
        else:
            sample_indices = [k for k in range(start_frame, end_frame+1)]
            add_idx = [random.choice(range(start_frame, end_frame+1)) for i in range(self.sample_num - frame_total)]#random.sample(range(start_frame, end_frame+1), self.sample_num - frame_total)
            sample_indices.extend(add_idx)
        sample_indices.sort()

        all_img_tensor = torch.zeros(0, self.input_size, self.input_size)

        if self.randomflip and random.random() < 0.5:
            horizon_flip = True
        else:
            horizon_flip = False

        for frame_idx in sample_indices:
            img_path = frame_dir + "/" + str(frame_idx).zfill(5) + self.img_ext

            img = Image.open(img_path)

            if self.transform:
                img = self.transform(img)

            all_img_tensor = torch.cat((all_img_tensor, img), dim=0)
        video_info = [frame_dir, str(start_frame), str(end_frame), str(label)]
        return all_img_tensor, label, "___".join(video_info)
