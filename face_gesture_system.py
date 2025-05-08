import cv2
import numpy as np
import os
import json
import time
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Set up logging with rotation to limit file size
log_file = 'gesture_system.log'
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Configure the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Clear any existing handlers
if root_logger.handlers:
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

# Create a rotating file handler (5 MB max size, keep 3 backup files)
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=5*1024*1024,  # 5 MB
    backupCount=3,
    encoding='utf-8'
)
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

class FaceGestureColorSystem:
    def __init__(self):
        logging.info("Initializing Face Gesture Color System")
        
        # Initialize the face cascade classifier
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            logging.info("Face cascade classifier loaded successfully")
        except Exception as e:
            logging.error(f"Error loading face cascade classifier: {str(e)}")
            raise
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logging.error("Failed to open webcam")
        else:
            logging.info("Webcam initialized successfully")
        
        # Color suggestions (RGB format)
        self.colors = [
            ('Red', (0, 0, 255)),
            ('Green', (0, 255, 0)),
            ('Blue', (255, 0, 0)),
            ('Yellow', (0, 255, 255)),
            ('Purple', (255, 0, 255)),
            ('Cyan', (255, 255, 0)),
            ('Orange', (0, 165, 255)),
            ('Pink', (147, 20, 255)),
            ('Teal', (128, 128, 0)),
            ('Lavender', (250, 230, 230))
        ]
        self.current_color_idx = 0
        
        # User preferences storage
        self.user_data_file = 'user_preferences.json'
        self.user_preferences = self.load_user_preferences()
        
        # State variables
        self.current_face_id = None
        self.face_features = {}  # All face features including temporary ones
        self.known_faces = []    # All known faces including temporary ones
        self.permanent_users = set()  # Set of user IDs that have been named and should be saved
        self.showing_message = False
        self.message_time = 0
        self.message = ""
        
        # Gesture detection area (bottom part of the screen)
        self.gesture_area_height = 150
        
        # Improved gesture detection state tracking
        self.static_gesture_count = 0
        self.last_gesture = 0
        self.gesture_cooldown = 10  # frames to wait before accepting new gesture
        self.last_gesture_time = 0
        self.last_gesture_processed = False  # Flag to track if we've already processed a gesture
        self.gesture_debug = True  # Print debug info to console
        self.last_color_change_time = 0
        self.color_change_cooldown = 1.0  # seconds
        
        # Load user face features from preferences
        self.load_known_faces()
        
    def load_user_preferences(self):
        if os.path.exists(self.user_data_file):
            logging.info(f"Loading user preferences from {self.user_data_file}")
            try:
                with open(self.user_data_file, 'r') as f:
                    data = json.load(f)
                    logging.info(f"Loaded {len(data)} user profiles")
                    return data
            except json.JSONDecodeError:
                logging.error(f"Error parsing {self.user_data_file} - invalid JSON format")
                return {}
        else:
            logging.info(f"User data file {self.user_data_file} not found, creating new preferences")
        return {}
    
    def save_user_preferences(self):
        # Create a copy of preferences containing only named users (permanent users)
        permanent_preferences = {}
        for user_id in self.permanent_users:
            if user_id in self.user_preferences:
                permanent_preferences[user_id] = self.user_preferences[user_id]
        
        # Save only the permanent users' data
        try:
            with open(self.user_data_file, 'w') as f:
                # Using indent=4 for proper formatting and sort_keys for consistent ordering
                json.dump(permanent_preferences, f, indent=4, sort_keys=True)
            logging.info(f"Saved {len(permanent_preferences)} permanent user profiles to {self.user_data_file} (formatted)")
            print(f"Saved {len(permanent_preferences)} permanent user profiles")
        except Exception as e:
            logging.error(f"Error saving user preferences: {str(e)}")
            print(f"Error saving user preferences: {str(e)}")
    
    def load_known_faces(self):
        """
        Load known face features from user preferences file
        Only users with explicitly named profiles are loaded
        """
        logging.info("Loading known face features from user profiles")
        for user_id, user_data in self.user_preferences.items():
            if 'face_features' in user_data and 'name' in user_data:
                # This is a named user, so add to permanent users set
                self.permanent_users.add(user_id)
                
                # Convert stored features back to numpy array
                features = np.array(user_data['face_features'])
                self.face_features[user_id] = features
                self.known_faces.append({
                    'id': user_id,
                    'features': features,
                    'name': user_data.get('name', f'User-{user_id[:5]}'),
                    'last_seen': user_data.get('last_seen', '')
                })
                print(f"Loaded face features for permanent user: {user_data.get('name', user_id)}")
    
    def extract_face_features(self, face_img):
        """
        Extract consistent features from a face image
        """
        # Normalize size
        face_resized = cv2.resize(face_img, (100, 100))
        
        # Convert to grayscale for more consistent features
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for lighting invariance
        face_eq = cv2.equalizeHist(face_gray)
        
        # Extract simple but effective features - HOG would be better with dlib
        # but we'll use simple gradient features for now
        gx = cv2.Sobel(face_eq, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(face_eq, cv2.CV_32F, 0, 1)
        
        # Compute gradient magnitude and direction
        mag, ang = cv2.cartToPolar(gx, gy)
        
        # Use magnitude-weighted orientation histogram as feature
        nbins = 16
        bins = np.linspace(0, 2*np.pi, nbins+1)
        h = np.zeros(nbins)
        for i in range(nbins):
            mask = (ang >= bins[i]) & (ang < bins[i+1])
            h[i] = np.sum(mag[mask])
        
        # Normalize
        if np.sum(h) > 0:
            h = h / np.sum(h)
        
        # Add some pixel intensity features for more robustness
        # Take average intensity in a 4x4 grid
        cells = 4
        cell_size = face_eq.shape[0] // cells
        intensity_features = []
        
        for i in range(cells):
            for j in range(cells):
                cell = face_eq[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
                intensity_features.append(np.mean(cell))
        
        # Combine features
        features = np.concatenate([h, np.array(intensity_features) / 255.0])
        return features
    
    def get_face_id(self, face_img):
        """
        More robust face identification based on face features
        Compares with known faces to find a match
        """
        # Extract features
        features = self.extract_face_features(face_img)
        logging.debug("Extracted face features for identification")
        
        # If we have known faces, try to match
        best_match_id = None
        best_match_score = float('inf')
        match_threshold = 0.4  # Smaller = stricter matching
        
        for known_face in self.known_faces:
            # Calculate distance (smaller = better match)
            dist = np.linalg.norm(features - known_face['features'])
            
            # If distance is below threshold and better than previous best match
            if dist < match_threshold and dist < best_match_score:
                best_match_score = dist
                best_match_id = known_face['id']
                logging.info(f"Identified existing user with ID {best_match_id}, similarity score: {dist:.5f}")
        
        # If no match found, create a new ID
        if best_match_id is None:
            # Create a new unique ID
            new_id = f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000)}"
            
            # Store the features
            self.face_features[new_id] = features
            
            # Add to known faces
            self.known_faces.append({
                'id': new_id,
                'features': features,
                'name': f'User-{new_id[-5:]}',
                'last_seen': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            print(f"Created new user with ID {new_id}")
            return new_id
        
        print(f"Identified existing user with ID {best_match_id}")
        return best_match_id
    
    def detect_gesture(self, frame):
        """
        Ultra-simplified gesture detection - uses simple motion detection and basic shape analysis
        Returns: 1 for thumbs up, -1 for thumbs down, 0 for no gesture
        """
        logging.debug("Starting gesture detection")
        # Define a region of interest for hand gestures (bottom half of the frame)
        height, width = frame.shape[:2]
        roi_height = height // 2
        roi = frame[height-roi_height:height, 0:width].copy()
        
        # Create a visualization copy
        viz = roi.copy()
        
        # Add instructions
        cv2.putText(viz, "Show thumbs UP/DOWN gesture here", (width//2-180, roi_height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(viz, "Or use 'u'/'d' keys on keyboard", (width//2-150, roi_height//2+40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
        
        # Simplest approach: Just look for any large enough movement
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)  # Heavy blur to reduce noise
        
        # Keep track of previous frames for motion detection
        if not hasattr(self, 'prev_gray'):
            self.prev_gray = gray
            small_viz = cv2.resize(viz, (width//2, roi_height//2))
            frame[0:small_viz.shape[0], 0:small_viz.shape[1]] = small_viz
            return 0
        
        # Calculate absolute difference between current and previous frame
        frame_delta = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray  # Update previous frame
        
        # Threshold the delta image
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate the thresholded image to fill in holes
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours in the threshold image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Show the motion areas
        motion_viz = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        viz_small = cv2.resize(motion_viz, (width//4, roi_height//4))
        viz[0:viz_small.shape[0], 0:viz_small.shape[1]] = viz_small
        
        # Check if there is significant motion
        motion_detected = False
        gesture_code = 0
        biggest_contour = None
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum size for consideration
                motion_detected = True
                if area > max_area:
                    max_area = area
                    biggest_contour = contour
        
        # Draw motion detection status
        status = "Motion Detected" if motion_detected else "No Motion"
        cv2.putText(viz, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if motion_detected else (0, 0, 255), 2)
        
        # If there's significant motion and we found a big contour
        if motion_detected and biggest_contour is not None:
            # Draw the biggest contour
            cv2.drawContours(viz, [biggest_contour], -1, (0, 255, 0), 2)
            
            # Get the bounding rectangle
            x, y, w, h = cv2.boundingRect(biggest_contour)
            cv2.rectangle(viz, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extremely simplified gesture detection based on motion location
            # We'll divide the ROI into top and bottom halves
            roi_mid_y = roi_height // 2
            
            # Calculate the center of the motion
            motion_y = y + h // 2
            
            # Draw a horizontal line at the middle of the ROI
            cv2.line(viz, (0, roi_mid_y), (width, roi_mid_y), (255, 255, 0), 1)
            
            # Show area and position
            cv2.putText(viz, f"Area: {max_area}", (x, y-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Simple rule: 
            # - If motion is mostly in top half = thumbs up
            # - If motion is mostly in bottom half = thumbs down
            if motion_y < roi_mid_y:  # Motion in top half = thumbs up
                cv2.putText(viz, "THUMBS UP!", (width//2 - 100, roi_height-50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.rectangle(viz, (0, 0), (width, roi_height), (0, 255, 0), 5)  # Green border
                print("\n*** THUMB UP DETECTED (simplified motion detection) ***\n")
                logging.info(f"Thumbs up gesture detected (area: {max_area}, position: {motion_y}/{roi_mid_y})")
                gesture_code = 1
            else:  # Motion in bottom half = thumbs down
                cv2.putText(viz, "THUMBS DOWN!", (width//2 - 120, roi_height-50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.rectangle(viz, (0, 0), (width, roi_height), (0, 0, 255), 5)  # Red border
                print("\n*** THUMB DOWN DETECTED (simplified motion detection) ***\n")
                logging.info(f"Thumbs down gesture detected (area: {max_area}, position: {motion_y}/{roi_mid_y})")
                gesture_code = -1
        
        # Show the ROI with visualization
        small_viz = cv2.resize(viz, (width//2, roi_height//2))
        frame[0:small_viz.shape[0], 0:small_viz.shape[1]] = small_viz
        
        return gesture_code
    
    def show_message(self, message, duration=2):
        self.showing_message = True
        self.message = message
        self.message_time = time.time()
        self.message_duration = duration
    
    def next_color(self):
        self.current_color_idx = (self.current_color_idx + 1) % len(self.colors)
        return self.colors[self.current_color_idx]
    
    def previous_color(self):
        self.current_color_idx = (self.current_color_idx - 1) % len(self.colors)
        return self.colors[self.current_color_idx]
    
    def run(self):
        logging.info("Starting Face Gesture Color System")
        print("Starting Face and Gesture Recognition System...")
        print("Press 'q' to quit")
        print("Show a thumbs up to like a color, thumbs down to see a different color")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logging.error("Failed to grab frame from webcam")
                print("Failed to grab frame")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Make a copy for display
            display_frame = frame.copy()
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Draw bottom gesture area guide
            height, width = frame.shape[:2]
            cv2.rectangle(display_frame, (0, height-self.gesture_area_height), 
                         (width, height), (50, 50, 50), 2)
            cv2.putText(display_frame, "Gesture Area", (width//2-60, height-self.gesture_area_height+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Process each detected face
            if len(faces) > 0:
                # For simplicity, focus on the largest face
                largest_face = max(faces, key=lambda rectangle: rectangle[2] * rectangle[3])
                x, y, w, h = largest_face
                
                # Draw rectangle around the face
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Extract the face image
                face_img = frame[y:y+h, x:x+w]
                
                # Get face ID
                self.current_face_id = self.get_face_id(face_img)
                
                # Initialize new user in preferences if not already exists
                if self.current_face_id not in self.user_preferences:
                    # Store the face features for future recognition
                    current_features = self.face_features.get(self.current_face_id)
                    
                    self.user_preferences[self.current_face_id] = {
                        "liked_colors": [],
                        "disliked_colors": [],
                        "last_seen": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        # Store features as a list since JSON can't store numpy arrays directly
                        "face_features": current_features.tolist() if current_features is not None else []
                    }
                    self.show_message("New user detected!")
                else:
                    # Update last seen
                    self.user_preferences[self.current_face_id]["last_seen"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Update face features if they've changed or aren't stored yet
                    current_features = self.face_features.get(self.current_face_id)
                    if current_features is not None and ("face_features" not in self.user_preferences[self.current_face_id] or 
                            len(self.user_preferences[self.current_face_id]["face_features"]) == 0):
                        self.user_preferences[self.current_face_id]["face_features"] = current_features.tolist()
                
                # Display current color suggestion with a more visually prominent design
                color_name, color_rgb = self.colors[self.current_color_idx]
                
                # Draw a larger, more noticeable color suggestion box
                suggestion_x = x + w + 20
                suggestion_y = y
                suggestion_width = 150
                suggestion_height = 80
                
                # Draw an outer border for the color suggestion
                cv2.rectangle(display_frame, 
                             (suggestion_x-5, suggestion_y-5), 
                             (suggestion_x+suggestion_width+5, suggestion_y+suggestion_height+5), 
                             (255, 255, 255), 2)
                
                # Draw the main color rectangle
                cv2.rectangle(display_frame, 
                             (suggestion_x, suggestion_y), 
                             (suggestion_x+suggestion_width, suggestion_y+suggestion_height), 
                             color_rgb, -1)
                
                # Add text label with drop shadow for better visibility
                # Shadow
                cv2.putText(display_frame, color_name, (suggestion_x+3, suggestion_y+40+3), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                # Text
                cv2.putText(display_frame, color_name, (suggestion_x+3, suggestion_y+40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
                # Add instruction text
                cv2.putText(display_frame, "Thumbs Up to Like", (suggestion_x, suggestion_y+suggestion_height+20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(display_frame, "Thumbs Down to Skip", (suggestion_x, suggestion_y+suggestion_height+40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Get user name if exists
                user_info = f"User: {self.current_face_id[:8]}..."
                if "name" in self.user_preferences[self.current_face_id]:
                    user_info = f"User: {self.user_preferences[self.current_face_id]['name']}"
                cv2.putText(display_frame, user_info, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Detect gestures
                gesture = self.detect_gesture(frame)
                
                # Handle gesture detection with more reliable response
                if gesture != 0:  # If a gesture is detected
                    if gesture == self.last_gesture:
                        self.static_gesture_count += 1
                    else:
                        self.static_gesture_count = 1
                        self.last_gesture = gesture
                    
                    # Only process gesture if it's been consistently detected for multiple frames
                    # and enough time has passed since last gesture action
                    current_time = time.time()
                    enough_time_passed = current_time - self.last_gesture_time > 0.8  # 0.8 second cooldown (reduced from 1.5)
                    
                    # Add visual feedback about gesture detection status
                    if self.static_gesture_count >= 1:
                        status_color = (0, 0, 255)  # Red while detecting
                        if self.static_gesture_count >= 2:  # Reduced from 3 to 2 frames
                            status_color = (0, 255, 0)  # Green when ready
                            
                        # Show gesture detection progress
                        cv2.rectangle(display_frame, (width-150, 10), (width-10, 40), (0, 0, 0), -1)
                        cv2.putText(display_frame, f"Gesture: {self.static_gesture_count}/2",  # Changed from 3 to 2
                                    (width-140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    
                    # Process gesture with less strict requirements - only need 2 consecutive frames
                    # Also add extra timestamp check to ensure we don't detect the same gesture repeatedly
                    current_time = time.time()
                    color_cooldown_passed = current_time - self.last_color_change_time > self.color_change_cooldown
                    
                    if self.static_gesture_count >= 2 and enough_time_passed and not self.last_gesture_processed and color_cooldown_passed:
                        # Set the processed flag to prevent multiple processing of the same gesture
                        self.last_gesture_processed = True
                        
                        # Store current color before potentially changing it
                        prev_color_idx = self.current_color_idx
                        prev_color_name = color_name
                        
                        # Force update the time to prevent multiple rapid color changes
                        self.last_color_change_time = current_time
                        
                        # Print confirmation that we're about to process a gesture
                        print(f"\n=== PROCESSING GESTURE {self.last_gesture} - CHANGING COLOR ===\n")
                        
                        # Process thumbs up gesture
                        if gesture == 1:  
                            # Add to liked colors if not already present
                            if color_name not in self.user_preferences[self.current_face_id]["liked_colors"]:
                                self.user_preferences[self.current_face_id]["liked_colors"].append(color_name)
                                # Remove from disliked if previously disliked
                                if color_name in self.user_preferences[self.current_face_id]["disliked_colors"]:
                                    self.user_preferences[self.current_face_id]["disliked_colors"].remove(color_name)
                            
                            # Provide clear feedback
                            self.show_message(f"👍 Added {color_name} to your preferences!")
                            
                            # Explicitly change to next color suggestion
                            next_color_name, _ = self.next_color()
                            
                            # Log for debugging
                            print(f"THUMBS UP CONFIRMED! Liked {prev_color_name}, changed to {next_color_name}")
                            
                            # Force update the display with new color
                            color_name, color_rgb = self.colors[self.current_color_idx]
                            
                            # Add a clear visual indicator that color has changed
                            cv2.rectangle(display_frame, (10, 10), (380, 50), (0, 0, 0), -1)
                            cv2.putText(display_frame, f"COLOR CHANGED: {prev_color_name} → {next_color_name}", 
                                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
                            # Reset gesture tracking to prevent duplicate detection
                            self.last_gesture_time = time.time()
                            self.static_gesture_count = 0
                        
                        # Process thumbs down gesture
                        elif gesture == -1:  
                            # Add to disliked colors if not already present
                            if color_name not in self.user_preferences[self.current_face_id]["disliked_colors"]:
                                self.user_preferences[self.current_face_id]["disliked_colors"].append(color_name)
                                # Remove from liked if previously liked
                                if color_name in self.user_preferences[self.current_face_id]["liked_colors"]:
                                    self.user_preferences[self.current_face_id]["liked_colors"].remove(color_name)
                            
                            # Provide clear feedback
                            self.show_message(f"👎 Skipping {color_name}, not your style!")
                            
                            # Explicitly change to next color suggestion
                            next_color_name, _ = self.next_color()
                            
                            # Log for debugging
                            print(f"THUMBS DOWN CONFIRMED! Disliked {prev_color_name}, changed to {next_color_name}")
                            
                            # Force update the display with new color
                            color_name, color_rgb = self.colors[self.current_color_idx]
                            
                            # Add a clear visual indicator that color has changed
                            cv2.rectangle(display_frame, (10, 10), (380, 50), (0, 0, 0), -1)
                            cv2.putText(display_frame, f"COLOR CHANGED: {prev_color_name} → {next_color_name}", 
                                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
                            # Reset gesture tracking to prevent duplicate detection
                            self.last_gesture_time = time.time()
                            self.static_gesture_count = 0
                else:
                    # Reset the processed flag when no gesture is detected
                    self.last_gesture_processed = False
                    
                    # Draw status indicator when cooldown is active
                    current_time = time.time()
                    time_since_last = current_time - self.last_gesture_time
                    if time_since_last < 1.5:  # If in cooldown period
                        cooldown_percent = min(1.0, time_since_last / 1.5)
                        bar_width = int(140 * cooldown_percent)
                        
                        # Show cooldown progress bar
                        cv2.rectangle(display_frame, (width-150, 50), (width-10, 70), (50, 50, 50), -1)
                        cv2.rectangle(display_frame, (width-150, 50), (width-150+bar_width, 70), (0, 255, 255), -1)
                        cv2.putText(display_frame, "Cooldown", (width-140, 90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            else:
                self.current_face_id = None
            
            # Display message if needed
            if self.showing_message:
                if time.time() - self.message_time < self.message_duration:
                    # Make message more visible with background
                    message_width = len(self.message) * 15
                    cv2.rectangle(display_frame, (width//2-message_width//2-10, 30), 
                                 (width//2+message_width//2+10, 70), (0, 0, 0), -1)
                    cv2.putText(display_frame, self.message, (width//2-message_width//2, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    self.showing_message = False
            
            # Show instructions with added keyboard shortcuts
            cv2.putText(display_frame, "Press 'u' for thumbs up (like color)", (10, height-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(display_frame, "Press 'd' for thumbs down (dislike color)", (10, height-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(display_frame, "Press 's' to save preferences", (10, height-50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, "Press 'n' to set name", (10, height-70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, "Press 'q' to quit", (10, height-90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
            # Display the frame
            cv2.imshow('Face and Gesture Recognition', display_frame)
            
            # Process keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            # Handle keyboard shortcuts
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_user_preferences()
                self.show_message("Preferences saved!")
            elif key == ord('n') and self.current_face_id:
                cv2.putText(display_frame, "Enter name in console", (width//2-150, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face and Gesture Recognition', display_frame)
                cv2.waitKey(1)
                name = input("Enter name for the current user: ")
                if name:
                    # Set name in preferences
                    self.user_preferences[self.current_face_id]["name"] = name
                    
                    # Mark this user as permanent (will be saved to file)
                    self.permanent_users.add(self.current_face_id)
                    
                    # Update the name in known_faces list too
                    for face in self.known_faces:
                        if face['id'] == self.current_face_id:
                            face['name'] = name
                            break
                    
                    # Save preferences immediately
                    self.save_user_preferences()
                    
                    logging.info(f"User {self.current_face_id} name set to '{name}' - marked as permanent")
                    self.show_message(f"Name set to {name}! User profile will be saved permanently.")
            # Keyboard shortcut for thumbs up (like current color)
            elif key == ord('u'):
                if self.current_face_id and time.time() - self.last_color_change_time > self.color_change_cooldown:
                    print("\n===== KEYBOARD THUMBS UP - LIKING CURRENT COLOR =====\n")
                    color_name, _ = self.colors[self.current_color_idx]
                    
                    # Add to liked colors list if not already there
                    if "liked_colors" not in self.user_preferences[self.current_face_id]:
                        self.user_preferences[self.current_face_id]["liked_colors"] = []
                    
                    if color_name not in self.user_preferences[self.current_face_id]["liked_colors"]:
                        self.user_preferences[self.current_face_id]["liked_colors"].append(color_name)
                        logging.info(f"User {self.user_preferences[self.current_face_id]['name']} liked color: {color_name}")
                    
                    # Remove from disliked if present
                    if "disliked_colors" in self.user_preferences[self.current_face_id] and \
                       color_name in self.user_preferences[self.current_face_id]["disliked_colors"]:
                        self.user_preferences[self.current_face_id]["disliked_colors"].remove(color_name)
                    
                    # Visual feedback for like
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (width, height), (0, 255, 0), -1)
                    cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
                    
                    # Show message and switch to next color
                    next_color_name, _ = self.next_color()
                    self.show_message(f"👍 LIKED: {color_name} → Now showing: {next_color_name}")
                    self.last_color_change_time = time.time()
            
            # Keyboard shortcut for thumbs down (dislike current color)
            elif key == ord('d'):
                if self.current_face_id and time.time() - self.last_color_change_time > self.color_change_cooldown:
                    print("\n===== KEYBOARD THUMBS DOWN - DISLIKING CURRENT COLOR =====\n")
                    color_name, _ = self.colors[self.current_color_idx]
                    
                    # Add to disliked colors list if not already there
                    if "disliked_colors" not in self.user_preferences[self.current_face_id]:
                        self.user_preferences[self.current_face_id]["disliked_colors"] = []
                    
                    if color_name not in self.user_preferences[self.current_face_id]["disliked_colors"]:
                        self.user_preferences[self.current_face_id]["disliked_colors"].append(color_name)
                        logging.info(f"User {self.user_preferences[self.current_face_id]['name']} disliked color: {color_name}")
                    
                    # Remove from liked if present
                    if "liked_colors" in self.user_preferences[self.current_face_id] and \
                       color_name in self.user_preferences[self.current_face_id]["liked_colors"]:
                        self.user_preferences[self.current_face_id]["liked_colors"].remove(color_name)
                    
                    # Visual feedback for dislike
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
                    
                    # Show message and switch to next color
                    next_color_name, _ = self.next_color()
                    self.show_message(f"👎 DISLIKED: {color_name} → Now showing: {next_color_name}")
                    self.last_color_change_time = time.time()
        
        # Clean up
        self.save_user_preferences()
        logging.info("User preferences saved")
        self.cap.release()
        cv2.destroyAllWindows()
        logging.info("Application closed")
        print("Application closed.")

if __name__ == "__main__":
    system = FaceGestureColorSystem()
    system.run()
