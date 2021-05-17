import os
import shutil

base_path = os.path.abspath('./')
# source_image_path = os.path.join(base_path, '../data_generator/Generate_tables/together/images')
# source_label_path = os.path.join(base_path, '../data_generator/Generate_tables/together/labels')
source_image_path = os.path.join(base_path, '../data_generator/Generate_tables_mixup/together/images')
source_label_path = os.path.join(base_path, '../data_generator/Generate_tables_mixup/together/labels')
out_image_path = os.path.join(base_path, 'data', 'ocr_table', 'images')
out_label_path = os.path.join(base_path, 'data', 'ocr_table', 'labels')
image_name_list = os.listdir(source_image_path)
count = 0
for image_name in image_name_list:
    if count % 30 == 0:
        shutil.copy(os.path.join(source_image_path, image_name), os.path.join(out_image_path, 'val', image_name))
        shutil.copy(os.path.join(source_label_path, image_name.replace('.jpg', '.txt')), os.path.join(out_label_path, 'val', image_name.replace('.jpg', '.txt')))
    elif count % 30 == 10:
        shutil.copy(os.path.join(source_image_path, image_name), os.path.join(out_image_path, 'test', image_name))
        shutil.copy(os.path.join(source_label_path, image_name.replace('.jpg', '.txt')), os.path.join(out_label_path, 'test', image_name.replace('.jpg', '.txt')))
    else:
        shutil.copy(os.path.join(source_image_path, image_name), os.path.join(out_image_path, 'train', image_name))
        shutil.copy(os.path.join(source_label_path, image_name.replace('.jpg', '.txt')), os.path.join(out_label_path, 'train', image_name.replace('.jpg', '.txt')))
    count += 1
    if count % 1000 == 0:
        print('processed num: ', count)
print('processed total num: ', count)
