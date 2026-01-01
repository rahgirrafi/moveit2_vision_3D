#!/usr/bin/env python3
"""
Button Press Vision Pipeline - Complete ROS 2 Node
Segments push-to-open button using SAM2 + Grounding DINO, computes 3D centroid, 
estimates surface normal, and executes button press using MoveIt2.
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import PointStamped, PoseStamped, Pose
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
import cv2
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
import tf2_ros
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation
from PIL import Image as PILImage

# MoveIt2 imports
from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.srv import GetPositionIK

# SAM2 + Grounding DINO imports
try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    SAM2_AVAILABLE = True
except ImportError as e:
    SAM2_AVAILABLE = False
    print(f"SAM2 + Grounding DINO not available: {e}")

class ButtonPressVisionNode(Node):
    def __init__(self):
        super().__init__('button_press_vision_node')
        
        # Parameters
        self.declare_parameter('camera_info_topic', '/sensors/d435_camera/camera_info')
        self.declare_parameter('rgb_topic', '/sensors/d435_camera/image')
        self.declare_parameter('depth_topic', '/sensors/d435_camera/depth_image')
        self.declare_parameter('pointcloud_topic', '/sensors/d435_camera/points')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'd435_color_optical_frame')
        self.declare_parameter('button_prompt', 'push button.')  # Must end with dot
        self.declare_parameter('detection_threshold', 0.25)
        self.declare_parameter('text_threshold', 0.3)
        self.declare_parameter('press_offset', 0.05)  # 5cm pre-press offset
        self.declare_parameter('press_depth', 0.02)   # 2cm press depth
        self.declare_parameter('sam2_checkpoint', 'vision_3d/resource/sam2.1_hiera_tiny.pt')
        self.declare_parameter('sam2_config', 'configs/sam2.1/sam2.1_hiera_t.yaml')
        self.declare_parameter('grounding_dino_model', 'IDEA-Research/grounding-dino-tiny')
        self.declare_parameter('show_opencv_windows', True)  # New parameter
        
        # State variables
        self.bridge = CvBridge()
        self.camera_info = None
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_pointcloud = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Visualization state
        self.show_windows = self.get_parameter('show_opencv_windows').value
        if self.show_windows:
            # cv2.namedWindow('RGB Input', cv2.WINDOW_NORMAL)
            # cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
            # cv2.namedWindow('Segmentation', cv2.WINDOW_NORMAL)
            # cv2.namedWindow('Depth Visualization', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Combined View', cv2.WINDOW_NORMAL)
            #adjust windpw size
            cv2.resizeWindow('Combined View', 1200, 800)
            self.get_logger().info('OpenCV windows created')
        
        # Subscribers
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            self.camera_info_callback,
            10
        )
        self.rgb_sub = self.create_subscription(
            Image,
            self.get_parameter('rgb_topic').value,
            self.rgb_callback,
            10
        )
        self.depth_sub = self.create_subscription(
            Image,
            self.get_parameter('depth_topic').value,
            self.depth_callback,
            10
        )
        self.pc_sub = self.create_subscription(
            PointCloud2,
            self.get_parameter('pointcloud_topic').value,
            self.pointcloud_callback,
            10
        )
        
        # Publishers for visualization
        self.centroid_pub = self.create_publisher(PointStamped, '/button/centroid', 10)
        self.normal_marker_pub = self.create_publisher(Marker, '/button/normal_marker', 10)
        self.segmentation_pub = self.create_publisher(Image, '/button/segmentation', 10)
        self.detection_pub = self.create_publisher(Image, '/button/detection', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/button/target_pose', 10)
        # Initialize SAM2 + Grounding DINO
        if SAM2_AVAILABLE:
            self.init_sam2_grounding_dino()
        
    
        
        # Timer for processing (1 Hz)
        self.create_timer(1.0, self.process_button_detection)
        
        self.get_logger().info('Button Press Vision Node initialized')
        if self.show_windows:
            self.get_logger().info('Press "q" in any OpenCV window to quit')
    
    def init_sam2_grounding_dino(self):
        """Initialize SAM2 and Grounding DINO models"""
        try:
            # Select device (use CPU fallback when CUDA isn't available)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
            # If CUDA is available, enable a few GPU optimizations
            if self.device == "cuda":
                try:
                    if torch.cuda.get_device_properties(0).major >= 8:
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                except Exception:
                    # Ignore device property checks if unavailable for some builds
                    pass

            # Initialize SAM2 image predictor
            sam2_checkpoint = self.get_parameter('sam2_checkpoint').value
            model_cfg = self.get_parameter('sam2_config').value
            sam2_model = build_sam2(model_cfg, sam2_checkpoint)
            print(f"SAM2 model built from {sam2_checkpoint} with config {model_cfg}")
            self.sam2_predictor = SAM2ImagePredictor(sam2_model)
            print(f"SAM2 model loaded on {self.device}")
            # Initialize Grounding DINO from HuggingFace on selected device
            model_id = self.get_parameter('grounding_dino_model').value
            self.grounding_processor = AutoProcessor.from_pretrained(model_id)
            self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id
            ).to(self.device)

            self.get_logger().info('SAM2 + Grounding DINO initialized successfully on ' + self.device)
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize SAM2 + Grounding DINO: {e}')
            self.sam2_predictor = None
            self.grounding_model = None
    
    
    def camera_info_callback(self, msg):
        """Store camera intrinsics"""
        self.camera_info = msg
    
    def rgb_callback(self, msg):
        """Store latest RGB image"""
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Show raw RGB
        if self.show_windows and self.latest_rgb is not None:
            display = self.latest_rgb.copy()
            cv2.putText(display, 'RGB Input', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.imshow('RGB Input', display)
            # cv2.waitKey(1)
    
    def depth_callback(self, msg):
        """Store latest depth image"""
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    
    def pointcloud_callback(self, msg):
        """Store latest point cloud"""
        self.latest_pointcloud = msg
    
    def visualize_depth(self, depth_image, mask=None):
        """Create depth visualization with optional mask overlay"""
        depth_viz = depth_image.copy()
        
        # Handle NaN and convert to display range
        depth_viz = np.nan_to_num(depth_viz, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize to 0-5m range
        depth_viz = np.clip(depth_viz, 0, 5.0)
        depth_viz = (depth_viz / 5.0 * 255).astype(np.uint8)
        
        # Apply colormap
        depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
        
        # Overlay mask if provided
        if mask is not None:
             # Create green overlay for mask - FIXED
            green_overlay = np.zeros_like(depth_viz)
            green_overlay[:, :] = [0, 255, 0]
            
            # Blend only where mask is True
            depth_viz[mask] = cv2.addWeighted(depth_viz[mask], 0.6, green_overlay[mask], 0.4, 0)
            
            # Draw mask outline
            contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(depth_viz, contours, -1, (255, 255, 255), 2)
        
        # Add scale bar
        cv2.rectangle(depth_viz, (10, depth_viz.shape[0]-30), (210, depth_viz.shape[0]-10), (0, 0, 0), -1)
        cv2.putText(depth_viz, '0m', (10, depth_viz.shape[0]-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(depth_viz, '5m', (180, depth_viz.shape[0]-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return depth_viz
    
    def segment_button(self, rgb_image):
        """
        Segment button using SAM2 + Grounding DINO
        Returns: binary mask (HxW) where button pixels are True, and bounding box
        """
        if not SAM2_AVAILABLE or self.sam2_predictor is None or self.grounding_model is None:
            self.get_logger().warn('SAM2 + Grounding DINO not available, using dummy mask')
            # Return dummy mask for testing (center region)
            h, w = rgb_image.shape[:2]
            mask = np.zeros((h, w), dtype=bool)
            mask[h//3:2*h//3, w//3:2*w//3] = True
            return mask, None
        
        try:
            # Convert BGR to RGB and then to PIL Image
            rgb_pil = PILImage.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
            
            # Step 1: Run Grounding DINO to detect button
            text_prompt = self.get_parameter('button_prompt').value
            # CRITICAL: Text must be lowercase and end with a dot
            if not text_prompt.endswith('.'):
                text_prompt += '.'
            text_prompt = text_prompt.lower()
            
            inputs = self.grounding_processor(
                images=rgb_pil, 
                text=text_prompt, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.grounding_model(**inputs)
            
            # Post-process detection results
            results = self.grounding_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.get_parameter('detection_threshold').value,
                text_threshold=self.get_parameter('text_threshold').value,
                target_sizes=[rgb_pil.size[::-1]]
            )
            
            if len(results[0]["boxes"]) == 0:
                self.get_logger().warn('No button detected by Grounding DINO')
                return None, None
            
            # Get the highest confidence detection
            input_boxes = results[0]["boxes"].cpu().numpy()
            labels = results[0]["labels"]
            scores = results[0]["scores"].cpu().numpy()
            
            best_idx = np.argmax(scores)
            best_box = input_boxes[best_idx]
            best_label = labels[best_idx]
            
            self.get_logger().info(f'Detected: {best_label} (confidence: {scores[best_idx]:.2f})')
            
            # Visualize detection (optional)
            self.visualize_detection(rgb_image, best_box, best_label, scores[best_idx])
            
            # Step 2: Run SAM2 image predictor to get precise mask
            self.sam2_predictor.set_image(np.array(rgb_pil.convert("RGB")))
            
            # Predict mask using the detected box
            masks, sam_scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=best_box[None, :],  # Add batch dimension
                multimask_output=False,
            )
            
            # Convert mask shape to (H, W)
            if masks.ndim == 3:
                mask = masks[0]  # Take first mask
            elif masks.ndim == 4:
                mask = masks[0, 0]  # Remove batch and mask dimensions
            else:
                mask = masks
            
            # Erode mask to reduce edge noise
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=2).astype(bool)
            
            self.get_logger().info(f'Segmentation successful. Mask size: {np.sum(mask)} pixels')
            
            return mask, best_box
            
        except Exception as e:
            self.get_logger().error(f'Segmentation failed: {e}')
            import traceback
            traceback.print_exc()
            return None, None
    
    def visualize_detection(self, image, box, label, score):
        """Visualize detection with bounding box"""
        vis_img = image.copy()
        x1, y1, x2, y2 = box.astype(int)
        
        # Draw box
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Add label background
        label_text = f"{label}: {score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(vis_img, (x1, y1-text_h-10), (x1+text_w+10, y1), (0, 255, 0), -1)
        cv2.putText(vis_img, label_text, (x1+5, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Publish
        detection_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding='bgr8')
        self.detection_pub.publish(detection_msg)
        
        return vis_img
    
    def compute_3d_centroid(self, mask, depth_image):
        """
        Compute 3D centroid from mask and depth image
        Fixed version with correct camera coordinate system
        Returns: (x, y, z) in camera frame, and filtered 3D points
        """
        if self.camera_info is None:
            self.get_logger().error('No camera info available')
            return None
        
        # Get camera intrinsics
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]
        
        self.get_logger().info(f'Camera intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}')
        
        # Get masked pixels (row, col) = (v, u) in image coordinates
        rows, cols = np.where(mask)
        
        if len(cols) == 0:
            self.get_logger().error('Mask is empty')
            return None
        
        self.get_logger().info(f'Mask contains {len(cols)} pixels')
        self.get_logger().info(f'  Pixel range - rows: [{rows.min()}, {rows.max()}], cols: [{cols.min()}, {cols.max()}]')
        
        # Get depth values at masked pixels
        depths = depth_image[rows, cols].astype(np.float32)
        #display on opencv
        
        
        # Handle NaN, Inf values (32FC1 encoding)
        depths = np.nan_to_num(depths, nan=0.0, posinf=0.0, neginf=0.0)
        # DIAGNOSTIC: Check depth statistics
        self.get_logger().info(f'Raw depth stats:')
        self.get_logger().info(f'  min={np.min(depths):.4f}, max={np.max(depths):.4f}')
        self.get_logger().info(f'  mean={np.mean(depths):.4f}, median={np.median(depths):.4f}')
        self.get_logger().info(f'  dtype: {depth_image.dtype}')
        
        # Count valid (non-zero) depths
        valid_mask = depths > 0
        valid_count = np.sum(valid_mask)
       
        
        self.get_logger().info(f'Valid depths: {valid_count}/{len(depths)} ({100*valid_count/len(depths):.1f}%)')
        
        if valid_count == 0:
            self.get_logger().error('❌ All depth values are zero or invalid!')
            self.get_logger().error('Possible causes:')
            self.get_logger().error('  1. RGB and depth are misaligned (use aligned depth topic)')
            self.get_logger().error('  2. Object too close (<0.3m) or too far (>3m) for D435')
            self.get_logger().error('  3. Surface is reflective/dark/transparent')
            return None
        
        # Get statistics of valid depths only
        valid_depths = depths[valid_mask]
        median_depth = np.median(valid_depths)
        mean_depth = np.mean(valid_depths)
        std_depth = np.std(valid_depths)
        
        self.get_logger().info(f'Valid depth statistics:')
        self.get_logger().info(f'  median={median_depth:.4f}m, mean={mean_depth:.4f}m, std={std_depth:.4f}m')
        
        # Sanity check for unit issues
        if median_depth < 0.01:
            self.get_logger().error(f'❌ Depth too small ({median_depth:.4f}m) - likely invalid data')
            return None
        elif median_depth > 100:
            self.get_logger().warn('⚠ Depth in millimeters detected, converting to meters')
            depths = depths / 1000.0
            median_depth = median_depth / 1000.0
            mean_depth = mean_depth / 1000.0
        
        # Adaptive depth filtering based on actual data
        # Use median ± 3*std as reasonable range
        if std_depth > 0:
            min_depth = max(0.1, median_depth - 3 * std_depth)
            max_depth = min(10.0, median_depth + 3 * std_depth)
        else:
            # Fallback if std is zero
            min_depth = max(0.1, median_depth * 0.5)
            max_depth = min(10.0, median_depth * 2.0)
        
        self.get_logger().info(f'Depth filtering range: [{min_depth:.3f}m, {max_depth:.3f}m]')
        
        # Apply depth filtering
        depth_valid = (depths > min_depth) & (depths < max_depth)
        filtered_count = np.sum(depth_valid)
        
        self.get_logger().info(f'After filtering: {filtered_count}/{len(depths)} points ({100*filtered_count/len(depths):.1f}%)')
        
        if filtered_count < 10:
            self.get_logger().error(f'❌ Too few valid points after filtering: {filtered_count}')
            self.get_logger().error('Solutions:')
            self.get_logger().error('  1. Use aligned depth topic')
            self.get_logger().error('  2. Move camera to optimal range (0.5-2m)')
            self.get_logger().error('  3. Ensure matte, non-reflective surface')
            return None
        
        # Filter arrays
        valid_rows = rows[depth_valid]
        valid_cols = cols[depth_valid]
        valid_depths = depths[depth_valid]
        
        # CRITICAL: Backproject to 3D using correct pinhole camera model

        #  X-right, Y-down, Z-forward (into scene)
        # 
        # Pinhole projection: [u, v, 1] = K * [X, Y, Z]
        #   u = fx * X/Z + cx
        #   v = fy * Y/Z + cy
        #
        # Inverse (backprojection):
        #   X = (u - cx) * Z / fx
        #   Y = (v - cy) * Z / fy
        #   Z = depth
        cx = 320
        cy = 240
        
        points_3d = np.zeros((len(valid_cols), 3), dtype=np.float32)
        
        # X = (col - cx) * depth / fx  (horizontal, right positive)
        points_3d[:, 0] = (valid_cols - cx) * valid_depths / fx
        
        # Y = (row - cy) * depth / fy  (vertical, down positive) 
        points_3d[:, 1] = (valid_rows - cy) * valid_depths / fy
        
        # Z = depth (forward into scene)
        points_3d[:, 2] = valid_depths
        
        
        
        # DIAGNOSTIC: Check 3D point cloud statistics
        self.get_logger().info(f'3D point cloud (camera frame):')
        self.get_logger().info(f'  X (right):   [{points_3d[:, 0].min():.4f}, {points_3d[:, 0].max():.4f}] m')
        self.get_logger().info(f'  Y (down):    [{points_3d[:, 1].min():.4f}, {points_3d[:, 1].max():.4f}] m')
        self.get_logger().info(f'  Z (forward): [{points_3d[:, 2].min():.4f}, {points_3d[:, 2].max():.4f}] m')
        
        # Statistical outlier removal (optional but recommended)
        if len(points_3d) >= 20:
            try:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_3d)
                
                # Adaptive parameters
                nb_neighbors = min(30, max(10, len(points_3d) // 5))
                std_ratio = 2.5  # Moderately lenient
                
                pcd_filtered, inlier_indices = pcd.remove_statistical_outlier(
                    nb_neighbors=nb_neighbors, 
                    std_ratio=std_ratio
                )
                
                points_3d_filtered = np.asarray(pcd_filtered.points)
                removed = len(points_3d) - len(points_3d_filtered)
                
                self.get_logger().info(f'Outlier removal: {removed} outliers removed, {len(points_3d_filtered)} points remain')
                
                # Only use filtered if we didn't remove too many
                if len(points_3d_filtered) >= 5:
                    points_3d = points_3d_filtered
                else:
                    self.get_logger().warn('Too many outliers removed, keeping original points')
                    
            except Exception as e:
                self.get_logger().warn(f'Outlier removal failed: {e}, using all points')
        else:
            self.get_logger().info(f'Only {len(points_3d)} points, skipping outlier removal')
        
        if len(points_3d) == 0:
            self.get_logger().error('❌ No points remaining after processing')
            return None
        
        # Compute centroid (mean of all 3D points)
        centroid = np.mean(points_3d, axis=0)
        centroid_std = np.std(points_3d, axis=0)
        #draw lines from (cx, cy) to centroid
        
        
        cv2.circle(self.latest_rgb, (int(cx), int(cy)), 5, (0,255,0), -1)
        cv2.line(self.latest_rgb, (int(cx), int(cy)), (int((centroid[0]*fx/centroid[2])+cx), int((centroid[1]*fy/centroid[2])+cy)), (255,0,0), 2)
        cv2.imwrite('centroid_line.png', self.latest_rgb)
        # VALIDATION: Check if centroid makes sense
        is_valid = True
        
        # Check if centroid is too close or too far
        distance = np.linalg.norm(centroid)
        if distance < 0.1:
            self.get_logger().error(f'❌ Centroid too close: {distance:.3f}m')
            is_valid = False
        elif distance > 5.0:
            self.get_logger().error(f'❌ Centroid too far: {distance:.3f}m')
            is_valid = False
        
        # Check if spread is reasonable
        if np.max(centroid_std) > 0.5:
            self.get_logger().warn(f'⚠ Large point spread: std={centroid_std}')
        
        if not is_valid:
            self.get_logger().error('Centroid validation failed!')
            return None
        
        # Success!
        self.get_logger().info('=' * 60)
        self.get_logger().info('✓ SUCCESS: 3D Centroid Computed')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'Position (camera frame):')
        self.get_logger().info(f'  X (right):   {centroid[0]:+.4f} m')
        self.get_logger().info(f'  Y (down):    {centroid[1]:+.4f} m')
        self.get_logger().info(f'  Z (forward): {centroid[2]:+.4f} m')
        self.get_logger().info(f'Distance from camera: {distance:.4f} m')
        self.get_logger().info(f'Valid points used: {len(points_3d)}')
        self.get_logger().info(f'Point spread (std): [{centroid_std[0]:.4f}, {centroid_std[1]:.4f}, {centroid_std[2]:.4f}]')
        self.get_logger().info('=' * 60)
        
        return centroid, points_3d
    
    def estimate_surface_normal(self, points_3d):
        """
        Estimate surface normal using Open3D
        Returns: normalized normal vector (3,)
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        
        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        
        # Orient normals towards camera (camera is at origin)
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))
        
        # Average normals for stability
        normals = np.asarray(pcd.normals)
        avg_normal = np.mean(normals, axis=0)
        avg_normal /= np.linalg.norm(avg_normal)
        
        # Normal should point towards camera (negative Z in camera frame for pressing away)
        # For button press, we want to press INTO the surface, so flip if needed
        if avg_normal[2] > 0:
            avg_normal *= -1
        
        return avg_normal
    
    def transform_to_base_frame(self, point_camera, normal_camera):
        """
        Transform point and normal from camera frame to base frame
        Returns: (point_base, normal_base)
        """
        try:
            camera_frame = self.get_parameter('camera_frame').value
            base_frame = self.get_parameter('base_frame').value
            self.get_logger().info(f"Transforming from {camera_frame} to {base_frame}")
        
            # Get transform
            transform = self.tf_buffer.lookup_transform(
                base_frame,
                camera_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0)
            )
            # LOG THE TRANSFORM
            t = transform.transform.translation
            r = transform.transform.rotation
            self.get_logger().info(f"TF Translation: [{t.x:.3f}, {t.y:.3f}, {t.z:.3f}]")
            self.get_logger().info(f"TF Rotation (quat): [{r.x:.3f}, {r.y:.3f}, {r.z:.3f}, {r.w:.3f}]")
            
            
            # Transform point
            point_stamped = PointStamped()
            point_stamped.header.frame_id = camera_frame
            point_stamped.point.x = float(point_camera[0])
            point_stamped.point.y = float(point_camera[1])
            point_stamped.point.z = float(point_camera[2])
            self.get_logger().info(f"Point before transform: [{point_camera[0]:.3f}, {point_camera[1]:.3f}, {point_camera[2]:.3f}]")

            point_base = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            self.get_logger().info(f"Point after transform: [{point_base.point.x:.3f}, {point_base.point.y:.3f}, {point_base.point.z:.3f}]")
            # Transform normal (as vector, not point)
            # Get rotation only
            q = transform.transform.rotation
            rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
            normal_base = rot.apply(normal_camera)
            
            return np.array([point_base.point.x, point_base.point.y, point_base.point.z]), normal_base
            
        except Exception as e:
            self.get_logger().error(f'Transform failed: {e}')
            return None, None
        
    def create_pose_from_point_normal(self, point, normal):
        """
        Create PoseStamped with Z-axis of camera_frame aligned to normal
        Returns: PoseStamped message
        """
        pose = PoseStamped()
        pose.header.frame_id = self.get_parameter('base_frame').value
        pose.header.stamp = self.get_clock().now().to_msg()
        
        # Set position
        pose.pose.position.x = float(point[0])
        pose.pose.position.y = float(point[1])
        pose.pose.position.z = float(point[2])
        
        # Create rotation matrix with Z-axis aligned to normal
        z_axis = normal / np.linalg.norm(normal)
        
        # Choose initial X-axis (avoid parallel to Z)
        x_axis = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(x_axis, z_axis)) > 0.9:
            x_axis = np.array([0.0, 1.0, 0.0])
        
        # Build orthonormal frame
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        
        # Create rotation matrix [X Y Z] as columns
        rot_matrix = np.column_stack([x_axis, y_axis, z_axis])
        rot = Rotation.from_matrix(rot_matrix)
        quat = rot.as_quat()  # [x, y, z, w]
        
        pose.pose.orientation.x = float(quat[0])
        pose.pose.orientation.y = float(quat[1])
        pose.pose.orientation.z = float(quat[2])
        pose.pose.orientation.w = float(quat[3])
        
        return pose
    
    def publish_visualizations(self, centroid_base, normal_base):
        """Publish visualization markers"""
        base_frame = self.get_parameter('base_frame').value
        
        # Publish centroid
        point_msg = PointStamped()
        point_msg.header.frame_id = base_frame
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.point.x = centroid_base[0]
        point_msg.point.y = centroid_base[1]
        point_msg.point.z = centroid_base[2]
        self.centroid_pub.publish(point_msg)
        
        # Publish normal as arrow
        marker = Marker()
        marker.header.frame_id = base_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = 0.01  # Shaft diameter
        marker.scale.y = 0.02  # Head diameter
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        
        # Arrow from centroid along normal
        marker.points = []
        start = Pose()
        start.position.x = centroid_base[0]
        start.position.y = centroid_base[1]
        start.position.z = centroid_base[2]
        
        end = Pose()
        end.position.x = centroid_base[0] + normal_base[0] * 0.1
        end.position.y = centroid_base[1] + normal_base[1] * 0.1
        end.position.z = centroid_base[2] + normal_base[2] * 0.1
        
        marker.points.append(start.position)
        marker.points.append(end.position)
        
        self.normal_marker_pub.publish(marker)
        
    
    def create_combined_visualization(self, rgb, detection, segmentation, depth):
        """Create 2x2 grid of all visualizations"""
        h, w = rgb.shape[:2]
        
        # Resize all to same size
        detection = cv2.resize(detection, (w, h))
        segmentation = cv2.resize(segmentation, (w, h))
        depth = cv2.resize(depth, (w, h))
        
        # Add labels
        cv2.putText(rgb, 'RGB', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(detection, 'Detection', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(segmentation, 'Segmentation', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(depth, 'Depth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Create grid
        top_row = np.hstack([rgb, detection])
        bottom_row = np.hstack([segmentation, depth])
        combined = np.vstack([top_row, bottom_row])
        
        return combined
    
    def create_pose_from_point_normal(self, point, normal):
        """Create PoseStamped with Z-axis aligned to normal"""
        pose = PoseStamped()
        pose.header.frame_id = self.get_parameter('base_frame').value
        pose.header.stamp = self.get_clock().now().to_msg()
        
        pose.pose.position.x = point[0]
        pose.pose.position.y = point[1]
        pose.pose.position.z = point[2]
        
        # Align Z-axis with normal
        z_axis = normal / np.linalg.norm(normal)
        x_axis = np.array([1, 0, 0])
        
        # Ensure orthogonality
        if abs(np.dot(x_axis, z_axis)) > 0.9:
            x_axis = np.array([0, 1, 0])
        
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        
        rot_matrix = np.column_stack([x_axis, y_axis, z_axis])
        rot = Rotation.from_matrix(rot_matrix)
        quat = rot.as_quat()
        
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]
        
        return pose
    
    def process_button_detection(self):
        """Main processing loop"""
        if self.latest_rgb is None or self.latest_depth is None:
            return
        
        self.get_logger().info('=' * 60)
        self.get_logger().info('Processing button detection...')
        
        # Step 1: Segment button using SAM2 + Grounding DINO
        result = self.segment_button(self.latest_rgb)
        if result[0] is None:
            self.get_logger().warn('Segmentation failed, skipping frame')
            return
        
        mask, bbox = result
        
        # Publish segmentation visualization
        seg_viz = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # Overlay on original image
        overlay = self.latest_rgb.copy()
        overlay[mask] = cv2.addWeighted(overlay[mask], 0.5, seg_viz[mask], 0.5, 0)
        # Draw contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        
        # if self.show_windows:
        #     cv2.imshow('Segmentation', overlay)
        
        seg_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        self.segmentation_pub.publish(seg_msg)
        
        # Visualize depth
        depth_viz = self.visualize_depth(self.latest_depth, mask)
        # if self.show_windows:
        #     cv2.imshow('Depth Visualization', depth_viz)
        
        result = self.compute_3d_centroid(mask, self.latest_depth)
        if result is None:
            self.get_logger().warn('Failed to compute centroid')
            if self.show_windows:
                cv2.waitKey(1)
            return
        
        centroid_camera, points_3d = result
        self.get_logger().info(f'Centroid (camera frame): [{centroid_camera[0]:.3f}, {centroid_camera[1]:.3f}, {centroid_camera[2]:.3f}]')
        
        # Step 3: Estimate surface normal
        normal_camera = self.estimate_surface_normal(points_3d)
        self.get_logger().info(f'Normal (camera frame): [{normal_camera[0]:.3f}, {normal_camera[1]:.3f}, {normal_camera[2]:.3f}]')
        
        # Step 4: Transform to base frame
        centroid_base, normal_base = self.transform_to_base_frame(centroid_camera, normal_camera)
        if centroid_base is None:
            self.get_logger().warn('Failed to transform to base frame')
            return
        
        self.get_logger().info(f'Centroid (base frame): [{centroid_base[0]:.3f}, {centroid_base[1]:.3f}, {centroid_base[2]:.3f}]')
        self.get_logger().info(f'Normal (base frame): [{normal_base[0]:.3f}, {normal_base[1]:.3f}, {normal_base[2]:.3f}]')
        target_pose = self.create_pose_from_point_normal(centroid_base, normal_base)
        self.pose_pub.publish(target_pose)
        
        self.get_logger().info('Published target pose:')
        self.get_logger().info(f'  Position: [{target_pose.pose.position.x:.3f}, '
                              f'{target_pose.pose.position.y:.3f}, '
                              f'{target_pose.pose.position.z:.3f}]')
        self.get_logger().info(f'  Orientation (quat): [{target_pose.pose.orientation.x:.3f}, '
                              f'{target_pose.pose.orientation.y:.3f}, '
                              f'{target_pose.pose.orientation.z:.3f}, '
                              f'{target_pose.pose.orientation.w:.3f}]')
        # Step 5: Publish visualizations
        self.publish_visualizations(centroid_base, normal_base)
        # Create combined view
        if self.show_windows:
            detection_img = self.latest_rgb.copy()
            if bbox is not None:
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(detection_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            combined = self.create_combined_visualization(
                self.latest_rgb,
                detection_img,
                overlay,
                depth_viz
            )
            cv2.imshow('Combined View', combined)
            
            # Check for quit
            key = cv2.waitKey(1)
            if key == ord('q'):
                self.get_logger().info('Quit requested')
                rclpy.shutdown()
        
        self.get_logger().info('Detection cycle complete!')
        self.get_logger().info('=' * 60)
        
        # Step 6: Execute button press (uncomment when ready)
        # success = self.execute_button_press(centroid_base, normal_base)
        
    def destroy_node(self):
        """Cleanup OpenCV windows"""
        if self.show_windows:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ButtonPressVisionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()