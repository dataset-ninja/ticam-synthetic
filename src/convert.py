import os
import shutil
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from cv2 import connectedComponents
from dataset_tools.convert import unpack_if_archive
from PIL import Image
from supervisely.io.fs import get_file_name, get_file_name_with_ext
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###
    rgb_images_path = "/home/alex/DATASETS/TODO/TICaM/Synthetic_images/RGB_wholeImage"
    grayscale_images_path = "/home/alex/DATASETS/TODO/TICaM/Synthetic_images/grayscale_wholeImage"
    depth_images_path = "/home/alex/DATASETS/TODO/TICaM/Synthetic_images/depthmaps_normalized"

    masks_path = "/home/alex/DATASETS/TODO/TICaM/Synthetic_labels/element_segmentations_wholeImage"
    bboxes_path = "/home/alex/DATASETS/TODO/TICaM/Synthetic_labels/boundingBoxes_wholeImage"

    batch_size = 10
    group_tag_name = "im id"
    images_ext = ".png"
    bboxes_ext = ".txt"
    ds_name = "train"

    def get_class_by_gray_pixel_value(gray_pixel_value):
        """
        Returns for a grayscale pixel value the corresponding ground truth label as an integer (0 to 4).
        Further, the function also outputs on which seat the object is placed (left or right).

        The relationship between gray_pixel_value and the class integer depends on the grayscale transformation function used.
        These functions should work fine with PIL and OpenCV.

        0 = background
        1 = infant seat
        2 = child seat
        3 = person
        4 = everyday object

        Keyword arguments:
        gray_pixel_value -- grayscale pixel value between 0 and 225

        Return:
        class_label, position
        """

        # background
        if gray_pixel_value == 226 or gray_pixel_value == 225:
            return 0, None

        # infant
        if gray_pixel_value == 173 or gray_pixel_value == 172:
            return 1, "left"

        # child
        if gray_pixel_value == 175 or gray_pixel_value == 174:
            return 2, "left"

        # person
        if gray_pixel_value == 29:
            return 3, "left"

        if gray_pixel_value == 132 or gray_pixel_value == 131:
            return 3, "right"

        # everyday objects
        if gray_pixel_value == 105:
            return 4, "left"

        # infant seat
        if gray_pixel_value == 76:
            return 5, "left"

        # child seat
        if gray_pixel_value == 150 or gray_pixel_value == 149:
            return 6, "left"

        return None, None

    def create_ann(image_path):
        labels = []
        tags = []

        group_tag = sly.Tag(group_tag_meta, value=get_file_name(image_path)[36:])
        tags.append(group_tag)

        left_value = image_path.split("_")[-3]
        left_value = gt_labels[int(left_value)]
        left = sly.Tag(left_meta, value=left_value)
        tags.append(left)

        right_value = get_file_name(image_path).split("_")[-1]
        right_value = gt_labels[int(right_value)]
        right = sly.Tag(right_meta, value=right_value)
        tags.append(right)

        img_height = 512
        img_wight = 512

        mask_path = os.path.join(masks_path, get_file_name_with_ext(image_path))
        mask_pil = Image.open(mask_path).convert("L")
        ann_np = np.asarray(mask_pil)
        unique_pixels = np.unique(ann_np)

        for pixel in unique_pixels:
            label_tags = []
            class_label_idx, position_value = get_class_by_gray_pixel_value(pixel)
            if class_label_idx == 0:
                continue
            if position_value is not None:
                position = sly.Tag(position_meta, value=position_value)
                label_tags.append(position)
            class_name = idx_to_class.get(class_label_idx)
            obj_class = meta.get_obj_class(class_name)
            mask = ann_np == pixel
            ret, curr_mask = connectedComponents(mask.astype("uint8"), connectivity=8)
            for i in range(1, ret):
                obj_mask = curr_mask == i
                curr_bitmap = sly.Bitmap(obj_mask)
                if curr_bitmap.area > 30:
                    curr_label = sly.Label(curr_bitmap, obj_class, tags=label_tags)
                    labels.append(curr_label)

        bbox_path = os.path.join(bboxes_path, get_file_name(image_path) + bboxes_ext)
        with open(bbox_path) as f:
            content = f.read().split("\n")
            for curr_data in content:
                if len(curr_data) > 0:
                    curr_bboxes_data = list(map(int, curr_data.split(",")))
                    class_name = idx_to_class[curr_bboxes_data[0]]
                    obj_class = meta.get_obj_class(class_name)
                    left = curr_bboxes_data[1]
                    right = curr_bboxes_data[3]
                    top = curr_bboxes_data[2]
                    bottom = curr_bboxes_data[4]
                    rectangle = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
                    label = sly.Label(rectangle, obj_class)
                    labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)

    activity_meta = sly.TagMeta("activity", sly.TagValueType.ANY_STRING)
    position_meta = sly.TagMeta("position", sly.TagValueType.ANY_STRING)
    left_meta = sly.TagMeta("left seat", sly.TagValueType.ANY_STRING)
    right_meta = sly.TagMeta("right seat", sly.TagValueType.ANY_STRING)

    gt_labels = {
        0: "empty seat",
        1: "infant in an infant seat",
        2: "child in a child seat",
        3: "person",
        4: "everyday objects",
        5: "empty infant seat",
        6: "empty child seat",
    }

    group_tag_meta = sly.TagMeta(group_tag_name, sly.TagValueType.ANY_STRING)
    meta = sly.ProjectMeta(tag_metas=[group_tag_meta, position_meta, left_meta, right_meta])

    idx_to_class = {
        1: "infant",
        2: "child",
        3: "person",
        4: "everyday object",
        5: "infant seat",
        6: "child seat",
    }

    for idx, name in idx_to_class.items():
        obj_class = sly.ObjClass(name, sly.AnyGeometry)
        meta = meta.add_obj_class(obj_class)

    api.project.update_meta(project.id, meta.to_json())
    api.project.images_grouping(id=project.id, enable=True, tag_name=group_tag_name)

    dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

    images_names = os.listdir(rgb_images_path)

    progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

    for img_names_batch in sly.batched(images_names, batch_size=batch_size):
        img_pathes_batch = []
        images_names_batch = []
        for im_name in img_names_batch:
            images_names_batch.append(im_name)
            im_path = os.path.join(rgb_images_path, im_name)
            img_pathes_batch.append(im_path)

            images_names_batch.append("grayscale_" + im_name)
            im_path = os.path.join(grayscale_images_path, im_name)
            img_pathes_batch.append(im_path)

            images_names_batch.append("depth_" + im_name)
            im_path = os.path.join(depth_images_path, im_name)
            img_pathes_batch.append(im_path)

        img_infos = api.image.upload_paths(dataset.id, images_names_batch, img_pathes_batch)
        img_ids = [im_info.id for im_info in img_infos]

        anns = []
        for i in range(0, len(img_pathes_batch), 3):
            ann = create_ann(img_pathes_batch[i])
            anns.extend([ann, ann, ann])

        api.annotation.upload_anns(img_ids, anns)

        progress.iters_done_report(len(img_names_batch))

    return project
