
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import keyboard
import sys
from queue import Queue, Empty
from dataclasses import dataclass
from typing import Optional, Tuple
from onvif import ONVIFCamera
import urllib.parse
import os

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the face mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=3,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define facial feature indices
# Eyes
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Nose
NOSE_TIP = 4
NOSE_BRIDGE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 12, 14, 15, 16, 17, 18, 200, 199, 175]

# Mouth
MOUTH_OUTER = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415]
MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

# Face outline
FACE_OUTLINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

@dataclass
class CameraConfig:
    """Configuration for the ONVIF camera"""
    host: str
    port: int
    username: str
    password: str
    degree_step: float = 0.1
    frame_buffer_size: int = 2
    max_queue_size: int = 10
    skip_frames: bool = True
    # Auto-tracking settings
    auto_tracking: bool = True
    tracking_zone_center: float = 0.5  # Center zone (0.4-0.6 = center 20% of frame)
    tracking_zone_margin: float = 0.1  # Margin around center zone
    tracking_sensitivity: float = 0.005  # Reduced tracking sensitivity for smaller movements
    max_tracking_speed: float = 2.0    # Maximum tracking speed multiplier

@dataclass
class PTZLimits:
    """PTZ movement limits"""
    pan_min: float
    pan_max: float
    tilt_min: float
    tilt_max: float

@dataclass
class PTZPosition:
    """Current PTZ position"""
    pan: float
    tilt: float

def draw_face_features(frame, face_landmarks):
    """Draw facial features on the frame"""
    h, w, _ = frame.shape
    
    # Draw face outline
    face_points = []
    for idx in FACE_OUTLINE:
        landmark = face_landmarks.landmark[idx]
        x, y = int(landmark.x * w), int(landmark.y * h)
        face_points.append([x, y])
    face_points = np.array(face_points, np.int32)
    cv2.polylines(frame, [face_points], True, (255, 0, 0), 2)
    
    # Draw eyes
    # Left eye
    left_eye_points = []
    for idx in LEFT_EYE:
        landmark = face_landmarks.landmark[idx]
        x, y = int(landmark.x * w), int(landmark.y * h)
        left_eye_points.append([x, y])
    left_eye_points = np.array(left_eye_points, np.int32)
    cv2.polylines(frame, [left_eye_points], True, (0, 255, 0), 2)
    
    # Right eye
    right_eye_points = []
    for idx in RIGHT_EYE:
        landmark = face_landmarks.landmark[idx]
        x, y = int(landmark.x * w), int(landmark.y * h)
        right_eye_points.append([x, y])
    right_eye_points = np.array(right_eye_points, np.int32)
    cv2.polylines(frame, [right_eye_points], True, (0, 255, 0), 2)
    
    # Draw nose
    nose_points = []
    for idx in NOSE_BRIDGE:
        landmark = face_landmarks.landmark[idx]
        x, y = int(landmark.x * w), int(landmark.y * h)
        nose_points.append([x, y])
    nose_points = np.array(nose_points, np.int32)
    cv2.polylines(frame, [nose_points], True, (0, 0, 255), 2)
    
    # Draw mouth
    # Outer mouth
    mouth_outer_points = []
    for idx in MOUTH_OUTER:
        landmark = face_landmarks.landmark[idx]
        x, y = int(landmark.x * w), int(landmark.y * h)
        mouth_outer_points.append([x, y])
    mouth_outer_points = np.array(mouth_outer_points, np.int32)
    cv2.polylines(frame, [mouth_outer_points], True, (255, 255, 0), 2)
    
    # Inner mouth
    mouth_inner_points = []
    for idx in MOUTH_INNER:
        landmark = face_landmarks.landmark[idx]
        x, y = int(landmark.x * w), int(landmark.y * h)
        mouth_inner_points.append([x, y])
    mouth_inner_points = np.array(mouth_inner_points, np.int32)
    cv2.polylines(frame, [mouth_inner_points], True, (255, 0, 255), 2)
    
    # Draw key points for better visualization
    for landmark in face_landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

def get_face_center(face_landmarks) -> Tuple[float, float]:
    """Get the center point of a face (normalized coordinates 0-1)"""
    # Use nose tip as face center
    nose_tip = face_landmarks.landmark[NOSE_TIP]
    return nose_tip.x, nose_tip.y

class ONVIFFaceDetectionController:
    """ONVIF camera controller with MediaPipe face detection and auto-tracking"""
    
    def __init__(self, config: CameraConfig):
        """Initialize the camera controller with configuration"""
        self.config = config
        self.camera = ONVIFCamera(
            host=config.host,
            port=config.port,
            user=config.username,
            passwd=config.password,
            wsdl_dir='C:/Users/vitch/OneDrive/Desktop/onvif_cam/wsdl'
        )
        self.media_service = self.camera.create_media_service()
        self.ptz_service = self.camera.create_ptz_service()
        self.media_profile = self.media_service.GetProfiles()[0]
        self.ptz_config = self._get_ptz_config()
        self.limits = self._get_ptz_limits()
        self.current_position = self._get_current_position()
        self.stream_url = self._get_stream_url()
        self.stream_thread: Optional[threading.Thread] = None
        self.process_thread: Optional[threading.Thread] = None
        self.is_streaming = False
        self.frame_queue = Queue(maxsize=config.max_queue_size)
        self.last_frame_time = 0
        self.frame_count = 0
        self.fps = 0
        self.face_count = 0
        
        # Auto-tracking variables
        self.last_face_center = None
        self.face_lost_frames = 0
        self.max_face_lost_frames = 30  # Stop tracking after 30 frames without face
        self.is_tracking = False
        self.tracking_thread: Optional[threading.Thread] = None

    def _get_ptz_config(self):
        """Get PTZ configuration options"""
        request = self.ptz_service.create_type('GetConfigurationOptions')
        request.ConfigurationToken = self.media_profile.PTZConfiguration.token
        return self.ptz_service.GetConfigurationOptions(request)

    def _get_ptz_limits(self) -> PTZLimits:
        """Get PTZ movement limits from camera"""
        pan_tilt_space = self.ptz_config.Spaces.AbsolutePanTiltPositionSpace[0]
        return PTZLimits(
            pan_min=pan_tilt_space.XRange.Min,
            pan_max=pan_tilt_space.XRange.Max,
            tilt_min=pan_tilt_space.YRange.Min,
            tilt_max=pan_tilt_space.YRange.Max
        )

    def _get_current_position(self) -> PTZPosition:
        """Get current PTZ position"""
        status = self.ptz_service.GetStatus({'ProfileToken': self.media_profile.token})
        return PTZPosition(
            pan=status.Position.PanTilt.x,
            tilt=status.Position.PanTilt.y
        )

    def _get_stream_url(self) -> str:
        """Get RTSP stream URL with authentication"""
        stream_uri = self.media_service.GetStreamUri({
            'StreamSetup': {
                'Stream': 'RTP-Unicast',
                'Transport': {
                    'Protocol': 'RTSP',
                    'Tunnel': None
                }
            },
            'ProfileToken': self.media_profile.token
        })
        
        parsed = urllib.parse.urlparse(stream_uri.Uri)
        stream_url = f"rtsp://{self.config.username}:{self.config.password}@{parsed.netloc}{parsed.path}"
        print(f"Stream URL: {stream_url}")
        return stream_url

    def _perform_absolute_move(self, pan: float, tilt: float) -> None:
        """Execute absolute PTZ movement"""
        request = self.ptz_service.create_type('AbsoluteMove')
        request.ProfileToken = self.media_profile.token
        request.Position = {'PanTilt': {'x': pan, 'y': tilt}}
        self.ptz_service.AbsoluteMove(request)
        self.current_position.pan = pan
        self.current_position.tilt = tilt

    def move_up(self) -> None:
        """Move camera up"""
        new_tilt = min(self.current_position.tilt + self.config.degree_step, self.limits.tilt_max)
        self._perform_absolute_move(self.current_position.pan, new_tilt)

    def move_down(self) -> None:
        """Move camera down"""
        new_tilt = max(self.current_position.tilt - self.config.degree_step, self.limits.tilt_min)
        self._perform_absolute_move(self.current_position.pan, new_tilt)

    def move_right(self) -> None:
        """Move camera right"""
        new_pan = min(self.current_position.pan + self.config.degree_step, self.limits.pan_max)
        self._perform_absolute_move(new_pan, self.current_position.tilt)

    def move_left(self) -> None:
        """Move camera left"""
        new_pan = max(self.current_position.pan - self.config.degree_step, self.limits.pan_min)
        self._perform_absolute_move(new_pan, self.current_position.tilt)

    def stop(self) -> None:
        """Stop all PTZ movement"""
        self.ptz_service.Stop({'ProfileToken': self.media_profile.token})

    def _auto_track_face(self, face_center: Tuple[float, float]) -> None:
        """Automatically track face by moving camera to keep face in center"""
        if not self.config.auto_tracking:
            return
            
        x, y = face_center
        
        # Calculate how far the face is from center
        center_x = 0.5
        center_y = 0.5
        
        # Define tracking zones
        zone_left = center_x - self.config.tracking_zone_center
        zone_right = center_x + self.config.tracking_zone_center
        zone_top = center_y - self.config.tracking_zone_center
        zone_bottom = center_y + self.config.tracking_zone_center
        
        # Check if face is outside the center zone
        pan_adjustment = 0
        tilt_adjustment = 0
        
        # Horizontal tracking (pan) - FOLLOW: camera moves to follow face
        if x < zone_left:
            # Face is too far left, move camera LEFT to follow face
            pan_adjustment = self.config.tracking_sensitivity
        elif x > zone_right:
            # Face is too far right, move camera RIGHT to follow face
            pan_adjustment = -self.config.tracking_sensitivity
            
        # Vertical tracking (tilt) - FOLLOW: camera moves to follow face
        if y < zone_top:
            # Face is too high, move camera UP to follow face
            tilt_adjustment = self.config.tracking_sensitivity
        elif y > zone_bottom:
            # Face is too low, move camera DOWN to follow face
            tilt_adjustment = -self.config.tracking_sensitivity
        
        # Apply movement if needed
        if abs(pan_adjustment) > 0.001 or abs(tilt_adjustment) > 0.001:
            new_pan = self.current_position.pan + pan_adjustment
            new_tilt = self.current_position.tilt + tilt_adjustment
            
            # Clamp to limits
            new_pan = max(self.limits.pan_min, min(self.limits.pan_max, new_pan))
            new_tilt = max(self.limits.tilt_min, min(self.limits.tilt_max, new_tilt))
            
            # Move camera
            self._perform_absolute_move(new_pan, new_tilt)

    def _capture_frames(self) -> None:
        """Capture frames from the camera and add them to the queue"""
        # Set RTSP transport to TCP for better reliability
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        cap = cv2.VideoCapture(self.stream_url)
        
        # Set OpenCV parameters for lower latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.frame_buffer_size)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("Error: Could not open video stream")
            return

        while self.is_streaming:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                time.sleep(0.1)
                continue

            # Skip frame if queue is full and skip_frames is enabled
            if self.frame_queue.full() and self.config.skip_frames:
                continue

            try:
                self.frame_queue.put(frame, block=False)
            except Queue.Full:
                continue

        cap.release()

    def _process_frames_with_face_detection(self) -> None:
        """Process frames with MediaPipe face detection and display them"""
        cv2.namedWindow('ONVIF Camera Face Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('ONVIF Camera Face Detection', 1280, 720)

        while self.is_streaming:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                
                # Calculate FPS
                current_time = time.time()
                self.frame_count += 1
                if current_time - self.last_frame_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_frame_time = current_time

                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame with MediaPipe
                results = face_mesh.process(rgb_frame)
                
                # Convert back to BGR for OpenCV
                frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                
                # Handle face detection and tracking
                self.face_count = 0
                face_detected = False
                
                if results.multi_face_landmarks:
                    self.face_count = len(results.multi_face_landmarks)
                    face_detected = True
                    
                    # Get the first (largest) face for tracking
                    primary_face = results.multi_face_landmarks[0]
                    face_center = get_face_center(primary_face)
                    
                    # Auto-track the face
                    if self.config.auto_tracking:
                        self._auto_track_face(face_center)
                        self.last_face_center = face_center
                        self.face_lost_frames = 0
                        self.is_tracking = True
                    
                    # Draw face features
                    for face_landmarks in results.multi_face_landmarks:
                        draw_face_features(frame, face_landmarks)
                        
                    # Draw tracking zone
                    h, w, _ = frame.shape
                    zone_left = int((0.5 - self.config.tracking_zone_center) * w)
                    zone_right = int((0.5 + self.config.tracking_zone_center) * w)
                    zone_top = int((0.5 - self.config.tracking_zone_center) * h)
                    zone_bottom = int((0.5 + self.config.tracking_zone_center) * h)
                    
                    cv2.rectangle(frame, (zone_left, zone_top), (zone_right, zone_bottom), 
                                (0, 255, 255), 2)
                    
                    # Draw face center point
                    center_x = int(face_center[0] * w)
                    center_y = int(face_center[1] * h)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                
                else:
                    # No face detected
                    self.face_lost_frames += 1
                    if self.face_lost_frames > self.max_face_lost_frames:
                        self.is_tracking = False
                        if self.config.auto_tracking:
                            self.stop()  # Stop camera movement when face is lost

                # Add info overlay
                cv2.putText(frame, f'FPS: {self.fps}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Faces: {self.face_count}', (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Camera: {self.config.host}', (10, frame.shape[0]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f'Pan: {self.current_position.pan:.1f}, Tilt: {self.current_position.tilt:.1f}', 
                           (10, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add tracking status
                tracking_status = "TRACKING" if self.is_tracking else "IDLE"
                tracking_color = (0, 255, 0) if self.is_tracking else (0, 0, 255)
                cv2.putText(frame, f'Status: {tracking_status}', (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, tracking_color, 2)
                
                # Add auto-tracking status
                auto_status = "AUTO-ON" if self.config.auto_tracking else "AUTO-OFF"
                cv2.putText(frame, f'Auto: {auto_status}', (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                cv2.imshow('ONVIF Camera Face Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Empty:
                continue

        cv2.destroyAllWindows()

    def start_streaming(self) -> None:
        """Start video streaming with face detection in separate threads"""
        if not self.is_streaming:
            self.is_streaming = True
            self.stream_thread = threading.Thread(target=self._capture_frames)
            self.process_thread = threading.Thread(target=self._process_frames_with_face_detection)
            self.stream_thread.daemon = True
            self.process_thread.daemon = True
            self.stream_thread.start()
            self.process_thread.start()

    def stop_streaming(self) -> None:
        """Stop video streaming"""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join()
        if self.process_thread:
            self.process_thread.join()
        cv2.destroyAllWindows()

    def increase_step_size(self) -> None:
        """Increase movement step size"""
        self.config.degree_step = min(self.config.degree_step + 1, 45)

    def decrease_step_size(self) -> None:
        """Decrease movement step size"""
        self.config.degree_step = max(self.config.degree_step - 1, 1)
        
    def toggle_auto_tracking(self) -> None:
        """Toggle auto-tracking on/off"""
        self.config.auto_tracking = not self.config.auto_tracking
        if not self.config.auto_tracking:
            self.stop()  # Stop movement when auto-tracking is disabled
        print(f"Auto-tracking: {'ON' if self.config.auto_tracking else 'OFF'}")

def main():
    """Main function for ONVIF camera face detection"""
    # Camera configuration - UPDATE THESE VALUES
    config = CameraConfig(
        host='192.168.1.2',      # Replace with your camera's IP address
        port=2020,               # Replace with your camera's port
        username='fradmin',      # Replace with your camera's username
        password='fradmin',      # Replace with your camera's password
        frame_buffer_size=2,
        max_queue_size=10,
        skip_frames=True,
        # Auto-tracking settings
        auto_tracking=True,      # Enable automatic face tracking
        tracking_zone_center=0.2,  # Center zone (20% of frame)
        tracking_sensitivity=0.009,  # Reduced tracking sensitivity for smaller movements
        max_tracking_speed=0.75,    # Maximum tracking speed
    )
    
    print("ONVIF Camera Face Detection with Auto-Tracking")
    print("=" * 50)
    print(f"Connecting to camera at {config.host}:{config.port}...")
    
    controller = ONVIFFaceDetectionController(config)
    controller.start_streaming()

    print("PTZ Camera Control with Auto Face Tracking")
    print("Use arrow keys to move the camera manually")
    print("Press 'q' to quit")
    print("Press 's' to stop movement")
    print("Press '+' to increase step size")
    print("Press '-' to decrease step size")
    print("Press 'a' to toggle auto-tracking")
    print("Press 'c' to save screenshot")
    print(f"Current step size: {config.degree_step} degrees")
    print(f"Auto-tracking: {'ON' if config.auto_tracking else 'OFF'}")

    def on_key_event(event):
        if event.event_type == keyboard.KEY_DOWN:
            if event.name == 'up':
                controller.move_up()
            elif event.name == 'down':
                controller.move_down()
            elif event.name == 'left':
                controller.move_left()
            elif event.name == 'right':
                controller.move_right()
            elif event.name == 's':
                controller.stop()
            elif event.name == 'q':
                print("Exiting...")
                controller.stop()
                controller.stop_streaming()
                sys.exit()
            elif event.name == '+':
                controller.increase_step_size()
                print(f"Step size increased to {config.degree_step} degrees")
            elif event.name == '-':
                controller.decrease_step_size()
                print(f"Step size decreased to {config.degree_step} degrees")
            elif event.name == 'a':
                controller.toggle_auto_tracking()
            elif event.name == 'c':
                # Save screenshot functionality can be added here
                print("Screenshot feature - implement as needed")

    keyboard.hook(on_key_event)
    keyboard.wait('q')

if __name__ == '__main__':
    main() 
