from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re

from .base_dataset import BaseImageDataset


class PersonX(BaseImageDataset):
    """
    PersonX
    Reference:
    Sun et al. Dissecting Person Re-identification from the Viewpoint of Viewpoint. CVPR 2019.

    Dataset statistics:
    # identities: 1266
    # images: 9840 (train) + 5136 (query) + 30816 (gallery)
    """
    dataset_dir = 'PersonX_v1'

    def __init__(self, root, verbose=True, **kwargs):
        super(PersonX, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dirs = [osp.join(self.dataset_dir, str(i), 'bounding_box_train') for i in range(4, 7)]
        self.query_dirs = [osp.join(self.dataset_dir, str(i), 'query') for i in range(4, 7)]
        self.gallery_dirs = [osp.join(self.dataset_dir, str(i), 'bounding_box_test') for i in range(4, 7)]

        self._check_before_run()

        train = self._process_dir(self.train_dirs, relabel=True)
        query = self._process_dir(self.query_dirs, relabel=False)
        gallery = self._process_dir(self.gallery_dirs, relabel=False)

        if verbose:
            print("=> PersonX loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, _ = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, _ = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, _ = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        for train_dir in self.train_dirs:
            if not osp.exists(train_dir):
                raise RuntimeError("'{}' is not available".format(train_dir))
        for query_dir in self.query_dirs:
            if not osp.exists(query_dir):
                raise RuntimeError("'{}' is not available".format(query_dir))
        for gallery_dir in self.query_dirs:
            if not osp.exists(gallery_dir):
                raise RuntimeError("'{}' is not available".format(gallery_dir))

    def _process_dir(self, dir_paths, relabel=False):
        img_paths = []
        for dir_path in dir_paths:
            img_paths.extend(glob.glob(osp.join(dir_path, '*.jpg')))
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')
        # cam2label = {3: 1, 4: 2, 8: 3, 10: 4, 11: 5, 12: 6}

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for idx, img_path in enumerate(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            # assert (camid in cam2label.keys())
            # camid = cam2label[camid]
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid, 0, idx))

        return dataset
