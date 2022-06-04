import pandas as pd
import os
import os.path as osp
import time

data_dir = '/opt/ml/input/id-photo-generator/data/clip_img'
mask_dir = '/opt/ml/input/id-photo-generator/data/matting'
csv_save_path = '/opt/ml/input/id-photo-generator/data/dataset.csv'

def valid_img_fname(fname, folder_id):
    if not fname.startswith(folder_id):
        return False

    if not fname.endswith('.jpg') and not fname.endswith('.png'):
        return False

    return True

def main():
    start_time = time.time()
    col_names = ['img_path', 'matting_path']
    df = pd.DataFrame(columns=col_names)

    img_cnt = 0
    save_path = csv_save_path.split('/')
    save_dir, save_fname = '/'.join(save_path[:-1]), save_path[-1]

    print('>>> Generating {} into {}'.format(save_fname, save_dir + '/'))
    for folder_id in os.listdir(data_dir): # 1803xxxxxx
        folder_id_path = osp.join(data_dir, folder_id)

        for clip_id in os.listdir(folder_id_path): # clip_xxxxxxxx
            if not clip_id.startswith('clip'):
                continue
            clip_id_path = osp.join(folder_id_path, clip_id)

            for img_fname in os.listdir(clip_id_path):
                if not valid_img_fname(img_fname, folder_id):
                    continue

                # img path
                img_path = osp.join(clip_id_path, img_fname)

                # mask path
                mask_path = img_path.replace('clip_img', 'matting')
                mask_path = mask_path.replace('clip', 'matting')
                mask_path = mask_path.replace('.jpg', '.png')

                # Append a row to DataFrame
                df.loc[len(df)] = [img_path, mask_path]

                img_cnt += 1

    df.to_csv(csv_save_path, index=False)
    end_time = time.time()

    print('>>> Saved into {}'.format(csv_save_path))
    print('Total number of data: {}'.format(img_cnt))
    print('Time taken: {} seconds'.format(round(end_time - start_time, 3)))


if __name__ == '__main__':
    main()
