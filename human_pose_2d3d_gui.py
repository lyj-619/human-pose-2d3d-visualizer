import cv2
import mediapipe as mp
import numpy as np
import math
import time
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os
import platform


class PoseEstimationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("2D/3D人体姿态估计实时可视化系统")
        self.root.geometry("1600x900")

        # 检测操作系统
        self.system = platform.system()

        # 初始化变量
        self.cap = None
        self.is_running = False
        self.show_landmarks = True
        self.show_connections = True
        self.show_angles = True
        self.record_data = False
        self.data_points = []
        self.last_processing_time = 0
        self.processing_interval = 0.05
        self.frame_count = 0
        self.start_time = time.time()

        # 3D显示参数
        self.scale_factor = 0.8
        self.center_point = [0.5, 0.5, 0.5]

        # 初始化MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 自定义绘制样式
        self.custom_landmark_style = mp.solutions.drawing_utils.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=3)
        self.custom_connection_style = mp.solutions.drawing_utils.DrawingSpec(
            color=(255, 165, 0), thickness=2)

        # 初始化姿态检测器
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 尝试加载中文字体
        self.font_path = None
        if self.system == "Windows":
            # Windows系统字体
            potential_fonts = [
                "C:/Windows/Fonts/simhei.ttf",  # 黑体
                "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
                "C:/Windows/Fonts/simsun.ttc",  # 宋体
            ]
        elif self.system == "Darwin":  # macOS
            potential_fonts = [
                "/System/Library/Fonts/PingFang.ttc",  # 苹方
                "/System/Library/Fonts/STHeiti Medium.ttc",  # 黑体
            ]
        else:  # Linux
            potential_fonts = [
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # 文泉驿微米黑
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Noto字体
            ]

        for font in potential_fonts:
            if os.path.exists(font):
                self.font_path = font
                break

        # 设置界面
        self.setup_ui()

        # 启动摄像头
        self.start_camera()

    def setup_ui(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 视频源选择
        source_frame = ttk.LabelFrame(control_frame, text="视频源", padding=5)
        source_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(source_frame, text="摄像头", command=self.use_camera,
                   style="Accent.TButton").pack(fill=tk.X, pady=2)
        ttk.Button(source_frame, text="选择视频文件", command=self.select_video_file,
                   style="Accent.TButton").pack(fill=tk.X, pady=2)

        # 显示选项
        display_frame = ttk.LabelFrame(control_frame, text="显示选项", padding=5)
        display_frame.pack(fill=tk.X, pady=(0, 10))

        self.landmarks_var = tk.BooleanVar(value=True)
        self.connections_var = tk.BooleanVar(value=True)
        self.angles_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(display_frame, text="显示关键点",
                        variable=self.landmarks_var,
                        command=self.toggle_landmarks).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(display_frame, text="显示连接线",
                        variable=self.connections_var,
                        command=self.toggle_connections).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(display_frame, text="显示角度",
                        variable=self.angles_var,
                        command=self.toggle_angles).pack(anchor=tk.W, pady=2)

        # 3D显示设置
        view_frame = ttk.LabelFrame(control_frame, text="3D显示设置", padding=5)
        view_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(view_frame, text="缩放比例:").pack(anchor=tk.W)
        self.scale_var = tk.DoubleVar(value=self.scale_factor)
        self.scale_scale = ttk.Scale(view_frame, from_=0.3, to=1.5,
                                     orient=tk.HORIZONTAL, variable=self.scale_var,
                                     command=self.update_scale)
        self.scale_scale.pack(fill=tk.X, pady=5)

        self.scale_label = ttk.Label(view_frame, text=f"{self.scale_factor:.1f}")
        self.scale_label.pack()

        ttk.Button(view_frame, text="自动调整", command=self.auto_adjust_scale,
                   style="Accent.TButton").pack(fill=tk.X, pady=2)

        # 数据记录
        data_frame = ttk.LabelFrame(control_frame, text="数据记录", padding=5)
        data_frame.pack(fill=tk.X, pady=(0, 10))

        self.record_btn = ttk.Button(data_frame, text="开始记录",
                                     command=self.toggle_recording,
                                     style="Accent.TButton")
        self.record_btn.pack(fill=tk.X, pady=2)

        ttk.Button(data_frame, text="保存数据", command=self.save_data,
                   style="Accent.TButton").pack(fill=tk.X, pady=2)
        ttk.Button(data_frame, text="清除数据", command=self.clear_data,
                   style="Accent.TButton").pack(fill=tk.X, pady=2)

        # 性能设置
        perf_frame = ttk.LabelFrame(control_frame, text="性能设置", padding=5)
        perf_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(perf_frame, text="处理间隔 (ms):").pack(anchor=tk.W)
        self.interval_scale = ttk.Scale(perf_frame, from_=10, to=200,
                                        orient=tk.HORIZONTAL,
                                        command=self.update_interval)
        self.interval_scale.set(50)
        self.interval_scale.pack(fill=tk.X, pady=5)

        self.interval_label = ttk.Label(perf_frame, text="50 ms")
        self.interval_label.pack()

        # 模型复杂度
        ttk.Label(perf_frame, text="模型复杂度:").pack(anchor=tk.W, pady=(5, 0))
        self.model_var = tk.IntVar(value=1)
        ttk.Radiobutton(perf_frame, text="轻量", variable=self.model_var,
                        value=0, command=self.change_model).pack(anchor=tk.W)
        ttk.Radiobutton(perf_frame, text="标准", variable=self.model_var,
                        value=1, command=self.change_model).pack(anchor=tk.W)
        ttk.Radiobutton(perf_frame, text="完整", variable=self.model_var,
                        value=2, command=self.change_model).pack(anchor=tk.W)

        # 控制按钮
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        self.start_btn = ttk.Button(btn_frame, text="开始检测",
                                    command=self.start_detection,
                                    style="Accent.TButton")
        self.start_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        ttk.Button(btn_frame, text="截图保存",
                   command=self.capture_screenshot,
                   style="Accent.TButton").pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 右侧显示区域
        display_main = ttk.Frame(main_frame)
        display_main.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 创建2D/3D并排显示的框架
        visualization_frame = ttk.Frame(display_main)
        visualization_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # 2D显示区域
        self.video_frame_2d = ttk.LabelFrame(visualization_frame, text="2D视图", padding=5)
        self.video_frame_2d.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.video_label_2d = ttk.Label(self.video_frame_2d, background='black')
        self.video_label_2d.pack(fill=tk.BOTH, expand=True)

        # 3D显示区域
        self.video_frame_3d = ttk.LabelFrame(visualization_frame, text="3D视图", padding=5)
        self.video_frame_3d.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # 初始化3D图形
        self.fig_3d = plt.figure(figsize=(8, 6))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')

        # 设置3D图形的初始视角
        self.ax_3d.view_init(elev=20, azim=-90)

        # 创建3D图形画布
        self.plot_canvas = FigureCanvasTkAgg(self.fig_3d, self.video_frame_3d)
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 3D视图控制按钮
        control_3d_frame = ttk.Frame(self.video_frame_3d)
        control_3d_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(control_3d_frame, text="重置视角",
                   command=self.reset_3d_view, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_3d_frame, text="前视图",
                   command=lambda: self.set_3d_view(elev=20, azim=-90), width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_3d_frame, text="侧视图",
                   command=lambda: self.set_3d_view(elev=20, azim=0), width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_3d_frame, text="俯视图",
                   command=lambda: self.set_3d_view(elev=90, azim=-90), width=10).pack(side=tk.LEFT, padx=2)

        # 底部信息面板
        info_frame = ttk.Frame(display_main)
        info_frame.pack(fill=tk.X, pady=(0, 5))

        # 状态信息
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(info_frame, textvariable=self.status_var,
                                 relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        # FPS显示
        self.fps_var = tk.StringVar(value="FPS: 0.0")
        fps_label = ttk.Label(info_frame, textvariable=self.fps_var,
                              width=10, relief=tk.SUNKEN)
        fps_label.pack(side=tk.RIGHT)

        # 角度显示区域
        angles_frame = ttk.LabelFrame(display_main, text="关节角度", padding=5)
        angles_frame.pack(fill=tk.X)

        # 创建角度显示网格
        angles_grid = ttk.Frame(angles_frame)
        angles_grid.pack(fill=tk.X, pady=5)

        self.angle_vars = {}
        angle_labels = [
            ("左肘", "left_elbow"),
            ("右肘", "right_elbow"),
            ("左膝", "left_knee"),
            ("右膝", "right_knee"),
            ("左肩", "left_shoulder"),
            ("右肩", "right_shoulder"),
            ("左髋", "left_hip"),
            ("右髋", "right_hip")
        ]

        for i, (label_text, key) in enumerate(angle_labels):
            frame = ttk.Frame(angles_grid)
            frame.grid(row=i // 2, column=i % 2, padx=10, pady=2, sticky=tk.W)

            ttk.Label(frame, text=f"{label_text}:").pack(side=tk.LEFT)
            self.angle_vars[key] = tk.StringVar(value="0.0°")
            ttk.Label(frame, textvariable=self.angle_vars[key],
                      width=8, font=('Arial', 10, 'bold')).pack(side=tk.LEFT)

        # 设置样式
        self.setup_styles()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

    def normalize_3d_coordinates(self, landmarks):
        """标准化3D坐标"""
        if landmarks is None or len(landmarks.landmark) < 1:
            return None

        # 提取原始坐标
        points = []
        for landmark in landmarks.landmark:
            # MediaPipe的坐标系：x向右，y向下，z向后
            # 转换为：x向右，y向后，z向上
            x = landmark.x
            y = -landmark.z
            z = 1.0 - landmark.y

            points.append([x, y, z])

        points = np.array(points)

        # 计算中心点
        if len(points) >= 24:
            center_indices = [11, 12, 23, 24]
            center_points = points[center_indices]
            center = np.mean(center_points, axis=0)
        else:
            center = np.mean(points, axis=0)

        # 中心化坐标
        centered_points = points - center

        # 计算包围盒大小
        max_range = np.max(np.abs(centered_points))
        if max_range < 0.001:
            max_range = 1.0

        # 缩放坐标
        scaled_points = centered_points / (max_range * 3.0) * self.scale_factor

        # 将坐标移动到中心点
        final_points = scaled_points + np.array(self.center_point)

        return final_points

    def add_text_to_image(self, image, text, position, font_size=20,
                          text_color=(255, 255, 255), bg_color=(0, 0, 0, 150)):

        # 将OpenCV图像转换为PIL图像
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image, 'RGBA')

        try:
            # 尝试使用中文字体
            if self.font_path:
                font = ImageFont.truetype(self.font_path, font_size)
            else:
                # 回退到默认字体
                font = ImageFont.load_default()
        except:
            # 如果字体加载失败，使用默认字体
            font = ImageFont.load_default()

        # 计算文本大小
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            # 如果计算文本大小失败，使用估计值
            text_width = len(text) * font_size
            text_height = font_size

        # 绘制背景矩形
        x, y = position
        padding = 5
        draw.rectangle([x - padding, y - padding,
                        x + text_width + padding, y + text_height + padding],
                       fill=bg_color)

        # 绘制文本
        draw.text((x, y), text, font=font, fill=text_color)

        # 转换回OpenCV图像
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def start_camera(self):
        """启动摄像头"""
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.is_running = True
            self.start_detection()
        except Exception as e:
            messagebox.showerror("错误", f"无法启动摄像头: {str(e)}")

    def start_detection(self):
        """开始检测"""
        if not self.is_running:
            self.is_running = True
            self.start_btn.config(text="停止检测")
            self.start_time = time.time()
            self.frame_count = 0
            self.update_frame()
        else:
            self.is_running = False
            self.start_btn.config(text="开始检测")

    def update_frame(self):
        """更新视频帧"""
        if not self.is_running or not self.cap or not self.cap.isOpened():
            return

        # 读取帧
        ret, frame = self.cap.read()
        if not ret:
            return

        # 计算处理间隔
        current_time = time.time()
        if current_time - self.last_processing_time > self.processing_interval:
            # 处理帧
            processed_frame, results = self.process_frame(frame)
            self.last_processing_time = current_time

            # 更新2D和3D显示
            self.update_2d_display(processed_frame, results)
            if results.pose_landmarks:
                normalized_points = self.normalize_3d_coordinates(results.pose_landmarks)
                if normalized_points is not None:
                    self.update_3d_display(results.pose_landmarks, normalized_points)

        # 计算FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            fps = self.frame_count / elapsed_time
            self.fps_var.set(f"FPS: {fps:.1f}")

        # 继续下一帧
        if self.is_running:
            self.root.after(1, self.update_frame)

    def process_frame(self, frame):
        """处理单帧图像"""
        # 转换颜色空间
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        # 姿态检测
        results = self.pose.process(frame_rgb)

        # 转换回BGR
        frame_rgb.flags.writeable = True
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 绘制2D结果
        if results.pose_landmarks:
            frame_bgr = self.draw_2d_pose(frame_bgr, results)

            # 计算并显示角度
            if self.show_angles:
                angles = self.calculate_angles(results.pose_landmarks, frame_bgr.shape)
                self.update_angle_display(angles)

            # 记录数据
            if self.record_data:
                self.record_pose_data(results.pose_landmarks)

        return frame_bgr, results

    def draw_2d_pose(self, image, results):
        """绘制2D姿态"""
        # 创建图像副本
        image_copy = image.copy()

        # 绘制关键点和连接线
        if self.show_connections:
            self.mp_drawing.draw_landmarks(
                image_copy,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.custom_landmark_style if self.show_landmarks else None,
                connection_drawing_spec=self.custom_connection_style
            )
        elif self.show_landmarks:
            self.mp_drawing.draw_landmarks(
                image_copy,
                results.pose_landmarks,
                landmark_drawing_spec=self.custom_landmark_style
            )

        return image_copy

    def update_2d_display(self, frame, results):
        """更新2D显示"""
        # 调整图像大小以适应显示区域
        display_size = (640, 480)
        frame_resized = cv2.resize(frame, display_size)

        # 添加检测状态信息
        if results.pose_landmarks:
            # 使用OpenCV添加英文文本（确保可显示）
            cv2.putText(frame_resized, "Pose Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 添加关键点数量信息
            num_landmarks = len(results.pose_landmarks.landmark)
            cv2.putText(frame_resized, f"Landmarks: {num_landmarks}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(frame_resized, "No Pose Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 添加FPS信息
        if hasattr(self, 'fps_var'):
            fps_text = self.fps_var.get()
            if fps_text.startswith("FPS: "):
                fps_value = fps_text[5:]
                cv2.putText(frame_resized, f"FPS: {fps_value}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 添加时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame_resized, timestamp, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 转换图像格式
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        # 更新标签
        self.video_label_2d.configure(image=img_tk)
        self.video_label_2d.image = img_tk

        # 更新状态
        if results.pose_landmarks:
            self.status_var.set(f"检测到姿态 | 关键点数: {len(results.pose_landmarks.landmark)}")
        else:
            self.status_var.set("未检测到姿态")

    def update_3d_display(self, landmarks, normalized_points):
        """更新3D图形显示"""
        if landmarks is None or normalized_points is None:
            return

        # 清除当前图形
        self.ax_3d.clear()

        # 提取坐标
        x_coords = normalized_points[:, 0]
        y_coords = normalized_points[:, 1]
        z_coords = normalized_points[:, 2]

        # 绘制3D点
        colors = []
        sizes = []

        for i in range(len(x_coords)):
            visibility = landmarks.landmark[i].visibility

            if visibility > 0.7:
                colors.append('red')
                sizes.append(50)
            elif visibility > 0.3:
                colors.append('orange')
                sizes.append(40)
            else:
                colors.append('gray')
                sizes.append(30)

        # 批量绘制关键点
        scatter = self.ax_3d.scatter(x_coords, y_coords, z_coords,
                                     c=colors, s=sizes, alpha=0.8, edgecolors='black')

        # 绘制连接线
        connections = self.mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(x_coords) and end_idx < len(x_coords):
                # 计算平均可见性
                avg_visibility = (landmarks.landmark[start_idx].visibility +
                                  landmarks.landmark[end_idx].visibility) / 2

                if avg_visibility > 0.7:
                    line_color = 'blue'
                    line_width = 2.5
                elif avg_visibility > 0.3:
                    line_color = 'cyan'
                    line_width = 2.0
                else:
                    line_color = 'lightgray'
                    line_width = 1.5

                # 绘制连接线
                self.ax_3d.plot([x_coords[start_idx], x_coords[end_idx]],
                                [y_coords[start_idx], y_coords[end_idx]],
                                [z_coords[start_idx], z_coords[end_idx]],
                                color=line_color, linewidth=line_width, alpha=0.8)

        # 设置坐标轴范围
        axis_limit = 0.5
        self.ax_3d.set_xlim([0.5 - axis_limit, 0.5 + axis_limit])
        self.ax_3d.set_ylim([0.5 - axis_limit, 0.5 + axis_limit])
        self.ax_3d.set_zlim([0.5 - axis_limit, 0.5 + axis_limit])

        # 设置坐标轴标签
        self.ax_3d.set_xlabel('X (Right)', fontsize=10)
        self.ax_3d.set_ylabel('Y (Front)', fontsize=10)
        self.ax_3d.set_zlabel('Z (Up)', fontsize=10)

        # 设置标题
        self.ax_3d.set_title('3D Human Pose Estimation', fontsize=12, fontweight='bold', pad=20)

        # 设置网格
        self.ax_3d.grid(True, alpha=0.3, linestyle='--')

        # 设置相等的轴比例
        self.ax_3d.set_box_aspect([1, 1, 1])

        # 添加坐标轴指示
        arrow_length = 0.1
        self.ax_3d.quiver(0.4, 0.5, 0.5, arrow_length, 0, 0, color='red',
                          arrow_length_ratio=0.1, linewidth=2)
        self.ax_3d.quiver(0.5, 0.4, 0.5, 0, arrow_length, 0, color='green',
                          arrow_length_ratio=0.1, linewidth=2)
        self.ax_3d.quiver(0.5, 0.5, 0.4, 0, 0, arrow_length, color='blue',
                          arrow_length_ratio=0.1, linewidth=2)

        # 添加坐标轴标签
        self.ax_3d.text(0.4 + arrow_length + 0.02, 0.5, 0.5, 'X',
                        fontsize=12, color='red', fontweight='bold')
        self.ax_3d.text(0.5, 0.4 + arrow_length + 0.02, 0.5, 'Y',
                        fontsize=12, color='green', fontweight='bold')
        self.ax_3d.text(0.5, 0.5, 0.4 + arrow_length + 0.02, 'Z',
                        fontsize=12, color='blue', fontweight='bold')

        # 添加可见度图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.8, edgecolor='black', label='High Visibility'),
            Patch(facecolor='orange', alpha=0.8, edgecolor='black', label='Medium Visibility'),
            Patch(facecolor='gray', alpha=0.8, edgecolor='black', label='Low Visibility')
        ]

        self.ax_3d.legend(handles=legend_elements, loc='upper right', fontsize=9)

        # 更新画布
        self.plot_canvas.draw()

    def reset_3d_view(self):
        """重置3D视图到默认视角"""
        self.ax_3d.view_init(elev=20, azim=-90)
        self.plot_canvas.draw()

    def set_3d_view(self, elev=20, azim=-90):
        """设置3D视图角度"""
        self.ax_3d.view_init(elev=elev, azim=azim)
        self.plot_canvas.draw()

    def update_scale(self, value):
        """更新缩放比例"""
        self.scale_factor = float(value)
        self.scale_label.config(text=f"{self.scale_factor:.1f}")

    def auto_adjust_scale(self):
        """自动调整缩放比例"""
        self.scale_factor = 0.8
        self.scale_var.set(self.scale_factor)
        self.scale_label.config(text=f"{self.scale_factor:.1f}")

    def calculate_angles(self, landmarks, image_shape):
        """计算关节角度"""
        angles = {}

        # 获取所有关键点坐标
        landmarks_array = []
        for landmark in landmarks.landmark:
            x = landmark.x * image_shape[1]
            y = landmark.y * image_shape[0]
            landmarks_array.append((x, y))

        # 计算各关节角度
        angle_points = {
            'left_elbow': [11, 13, 15],
            'right_elbow': [12, 14, 16],
            'left_knee': [23, 25, 27],
            'right_knee': [24, 26, 28],
            'left_shoulder': [13, 11, 23],
            'right_shoulder': [14, 12, 24],
            'left_hip': [11, 23, 25],
            'right_hip': [12, 24, 26]
        }

        for angle_name, indices in angle_points.items():
            if all(idx < len(landmarks_array) for idx in indices):
                p1 = landmarks_array[indices[0]]
                p2 = landmarks_array[indices[1]]
                p3 = landmarks_array[indices[2]]

                angle = self.calculate_angle_degrees(p1, p2, p3)
                angles[angle_name] = angle

        return angles

    def calculate_angle_degrees(self, a, b, c):
        """计算三点形成的角度（度数）"""
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])

        dot_product = ba[0] * bc[0] + ba[1] * bc[1]
        norm_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
        norm_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

        if norm_ba * norm_bc == 0:
            return 0

        cos_angle = dot_product / (norm_ba * norm_bc)
        cos_angle = max(-1, min(1, cos_angle))
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)

        return angle_deg

    def update_angle_display(self, angles):
        """更新角度显示"""
        for key, var in self.angle_vars.items():
            if key in angles:
                var.set(f"{angles[key]:.1f}°")

    def record_pose_data(self, landmarks):
        """记录姿态数据"""
        timestamp = datetime.now()
        landmarks_data = []

        for i, landmark in enumerate(landmarks.landmark):
            landmarks_data.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })

        self.data_points.append({
            'timestamp': timestamp,
            'landmarks': landmarks_data
        })

    def toggle_recording(self):
        """切换记录状态"""
        self.record_data = not self.record_data
        if self.record_data:
            self.record_btn.config(text="停止记录")
            self.status_var.set("正在记录数据...")
        else:
            self.record_btn.config(text="开始记录")
            self.status_var.set(f"记录停止，已保存 {len(self.data_points)} 个数据点")

    def save_data(self):
        """保存数据到文件"""
        if not self.data_points:
            messagebox.showwarning("警告", "没有可保存的数据")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'wb') as f:
                    pickle.dump(self.data_points, f)
                messagebox.showinfo("成功", f"数据已保存到: {filename}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")

    def clear_data(self):
        """清除数据"""
        if self.data_points and messagebox.askyesno("确认", "确定要清除所有数据吗？"):
            self.data_points = []
            self.status_var.set("数据已清除")

    def capture_screenshot(self):
        """保存截图"""
        if self.cap is None or not self.is_running:
            messagebox.showwarning("警告", "没有可用的视频流")
            return

        ret, frame = self.cap.read()
        if ret:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
            )

            if filename:
                # 处理2D姿态并保存
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)

                if results.pose_landmarks:
                    frame_with_pose = self.draw_2d_pose(frame, results)
                    cv2.imwrite(filename, frame_with_pose)
                else:
                    cv2.imwrite(filename, frame)

                messagebox.showinfo("成功", f"截图已保存到: {filename}")

    def use_camera(self):
        """使用摄像头"""
        if self.cap:
            self.cap.release()

        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.status_var.set("切换到摄像头")
        except Exception as e:
            messagebox.showerror("错误", f"无法打开摄像头: {str(e)}")

    def select_video_file(self):
        """选择视频文件"""
        filename = filedialog.askopenfilename(
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv"),
                ("所有文件", "*.*")
            ]
        )

        if filename:
            if self.cap:
                self.cap.release()

            try:
                self.cap = cv2.VideoCapture(filename)
                self.status_var.set(f"加载视频: {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("错误", f"无法打开视频文件: {str(e)}")

    def toggle_landmarks(self):
        """切换关键点显示"""
        self.show_landmarks = self.landmarks_var.get()

    def toggle_connections(self):
        """切换连接线显示"""
        self.show_connections = self.connections_var.get()

    def toggle_angles(self):
        """切换角度显示"""
        self.show_angles = self.angles_var.get()

    def update_interval(self, value):
        """更新处理间隔"""
        self.processing_interval = float(value) / 1000.0
        self.interval_label.config(text=f"{int(float(value))} ms")

    def change_model(self):
        """更改模型复杂度"""
        model_complexity = self.model_var.get()
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def on_closing(self):
        """关闭窗口时的清理工作"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        if self.pose:
            self.pose.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = PoseEstimationApp(root)

    # 设置关闭事件处理
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # 启动主循环
    root.mainloop()


if __name__ == "__main__":
    main()
