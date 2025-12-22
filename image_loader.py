"""
图像栈加载器模块
负责从文件夹中加载图像栈并生成缩略图
"""

import os
import cv2
import numpy as np
from PyQt6.QtGui import QPixmap, QImage
from typing import List, Tuple, Optional


class ImageStackLoader:
    """图像栈加载器"""
    
    # 支持的图像格式
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 支持的视频格式
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    
    def __init__(self):
        self.image_paths = []
        self.images = []
        self.thumbnail_size = (600, 400)  # 缩略图尺寸
    
    def load_from_folder(self, folder_path: str, scale_factor: float = 1.0) -> Tuple[bool, str, List[np.ndarray], List[str]]:
        """
        从文件夹加载图像栈
        
        Args:
            folder_path: 文件夹路径
            scale_factor: 下采样比例 (0.0 - 1.0)
            
        Returns:
            (成功标志, 消息, 图像列表, 文件名列表)
        """
        if not os.path.isdir(folder_path):
            return False, "Selected path is not a valid directory", [], []
        
        # 扫描文件夹中的所有图像文件
        image_files = []
        for filename in os.listdir(folder_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in self.SUPPORTED_FORMATS:
                full_path = os.path.join(folder_path, filename)
                image_files.append((filename, full_path))
        
        if not image_files:
            return False, "No supported image files found in the folder", [], []
        
        # 按文件名排序
        image_files.sort(key=lambda x: x[0])
        
        # 加载图像
        loaded_images = []
        filenames = []
        failed_count = 0
        
        for filename, full_path in image_files:
            try:
                img = cv2.imread(full_path)
                if img is not None:
                    # 如果需要下采样
                    if scale_factor != 1.0 and 0 < scale_factor < 1.0:
                        width = int(img.shape[1] * scale_factor)
                        height = int(img.shape[0] * scale_factor)
                        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                    
                    loaded_images.append(img)
                    filenames.append(filename)
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                print(f"Failed to load image {filename}: {e}")
        
        if not loaded_images:
            return False, "Could not load any image files", [], []
        
        self.images = loaded_images
        self.image_paths = [f[1] for f in image_files[:len(loaded_images)]]
        
        message = f"Loaded {len(loaded_images)} image(s)"
        if failed_count > 0:
            message += f" (failed: {failed_count})"
        
        return True, message, loaded_images, filenames

    def load_from_video(self, video_path: str, scale_factor: float = 1.0) -> Tuple[bool, str, List[np.ndarray], List[str]]:
        """
        从视频文件加载图像栈（每一帧作为一张图片）
        
        Args:
            video_path: 视频文件路径
            scale_factor: 下采样比例 (0.0 - 1.0)
            
        Returns:
            (成功标志, 消息, 图像列表, 文件名列表)
        """
        if not os.path.isfile(video_path):
            return False, "Video file does not exist", [], []
        
        ext = os.path.splitext(video_path)[1].lower()
        if ext not in self.SUPPORTED_VIDEO_FORMATS:
            return False, f"Unsupported video format: {ext}", [], []
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Failed to open video file", [], []
        
        loaded_images = []
        filenames = []
        frame_index = 0
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 如果需要下采样
                if scale_factor != 1.0 and 0 < scale_factor < 1.0:
                    width = int(frame.shape[1] * scale_factor)
                    height = int(frame.shape[0] * scale_factor)
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                
                loaded_images.append(frame)
                # 生成帧文件名，格式：视频名_frame_0001.png
                filenames.append(f"{video_name}_frame_{frame_index:04d}.png")
                frame_index += 1
        finally:
            cap.release()
        
        if not loaded_images:
            return False, "No frames could be extracted from the video", [], []
        
        self.images = loaded_images
        self.image_paths = [video_path] * len(loaded_images)  # 所有帧都来自同一个视频
        
        message = f"Loaded {len(loaded_images)} frame(s) from video"
        if total_frames > 0 and len(loaded_images) < total_frames:
            message += f" (expected: {total_frames})"
        
        return True, message, loaded_images, filenames
    
    def is_video_file(self, filepath: str) -> bool:
        """
        检查文件是否为支持的视频格式
        
        Args:
            filepath: 文件路径
            
        Returns:
            是否为视频文件
        """
        ext = os.path.splitext(filepath)[1].lower()
        return ext in self.SUPPORTED_VIDEO_FORMATS

    def load_from_filepaths(self, filepaths: list[str], scale_factor: float = 1.0) -> Tuple[bool, str, List[np.ndarray], List[str]]:
        """
        从给定的文件路径列表加载图像（保持原有接口返回格式）

        Args:
            filepaths: 有序的文件路径列表
            scale_factor: 下采样比例

        Returns:
            (成功标志, 消息, 图像列表, 文件名列表)
        """
        if not filepaths:
            return False, "No file paths provided", [], []

        loaded_images = []
        filenames = []
        failed_count = 0

        for full_path in filepaths:
            try:
                if not os.path.exists(full_path):
                    failed_count += 1
                    continue

                img = cv2.imread(full_path)
                if img is None:
                    failed_count += 1
                    continue

                # 如果需要下采样
                if scale_factor != 1.0 and 0 < scale_factor < 1.0:
                    width = int(img.shape[1] * scale_factor)
                    height = int(img.shape[0] * scale_factor)
                    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

                loaded_images.append(img)
                filenames.append(os.path.basename(full_path))
            except Exception as e:
                failed_count += 1
                print(f"Failed to load image {full_path}: {e}")

        if not loaded_images:
            return False, "Could not load any image files", [], []

        self.images = loaded_images
        self.image_paths = list(filepaths[: len(loaded_images)])

        message = f"Loaded {len(loaded_images)} image(s)"
        if failed_count > 0:
            message += f" (failed: {failed_count})"

        return True, message, loaded_images, filenames
    
    def create_pixmaps(self, images: List[np.ndarray], max_size: Tuple[int, int] = (800, 600)) -> List[QPixmap]:
        """
        将OpenCV图像转换为QPixmap列表（用于显示）
        
        Args:
            images: OpenCV图像列表 (BGR格式)
            max_size: 最大显示尺寸
            
        Returns:
            QPixmap列表
        """
        pixmaps = []
        for img in images:
            pixmap = self._cv_to_pixmap(img, max_size)
            pixmaps.append(pixmap)
        return pixmaps
    
    def create_thumbnails(self, images: List[np.ndarray], thumb_size: int = 40) -> List[QPixmap]:
        """
        创建缩略图列表
        
        Args:
            images: OpenCV图像列表
            thumb_size: 缩略图尺寸
            
        Returns:
            QPixmap缩略图列表
        """
        thumbnails = []
        for img in images:
            # 等比例缩放
            h, w = img.shape[:2]
            scale = thumb_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            pixmap = self._cv_to_pixmap(resized, (thumb_size, thumb_size))
            thumbnails.append(pixmap)
        return thumbnails
    
    def _cv_to_pixmap(self, cv_img: np.ndarray, max_size: Optional[Tuple[int, int]] = None) -> QPixmap:
        """
        将OpenCV图像转换为QPixmap
        
        Args:
            cv_img: OpenCV图像 (BGR格式)
            max_size: 最大尺寸 (width, height)，None表示不缩放
            
        Returns:
            QPixmap对象
        """
        # BGR转RGB
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        # 如果需要缩放
        if max_size is not None:
            h, w = rgb_img.shape[:2]
            max_w, max_h = max_size
            
            # 计算缩放比例
            scale = min(max_w / w, max_h / h)
            if scale < 1.0:  # 只在图像大于max_size时缩放
                new_w = int(w * scale)
                new_h = int(h * scale)
                rgb_img = cv2.resize(rgb_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        
        # 创建QImage
        q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # 转换为QPixmap
        return QPixmap.fromImage(q_img)
    
    def get_image_info(self, index: int) -> dict:
        """
        获取指定索引图像的信息
        
        Args:
            index: 图像索引
            
        Returns:
            包含图像信息的字典
        """
        if not self.images or index < 0 or index >= len(self.images):
            return {}
        
        img = self.images[index]
        path = self.image_paths[index] if index < len(self.image_paths) else ""
        
        return {
            'index': index,
            'path': path,
            'filename': os.path.basename(path) if path else "",
            'shape': img.shape,
            'size': f"{img.shape[1]}x{img.shape[0]}",
            'channels': img.shape[2] if len(img.shape) > 2 else 1,
            'dtype': img.dtype
        }
