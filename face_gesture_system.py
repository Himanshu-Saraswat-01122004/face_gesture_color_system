import cv2
import numpy as np
import os
import json
import time
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler


log_file = 'gesture_system.log'
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)


if root_logger.handlers:
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)


file_handler = RotatingFileHandler(
    log_file,
    maxBytes=5*1024*1024,
    backupCount=3,
    encoding='utf-8'
)
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

class FaceGestureColorSystem:
    def __init__(self):
        logging.info("Initializing Face Gesture Color System")
        

        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            logging.info("Face cascade classifier loaded successfully")
        except Exception as e:
            logging.error(f"Error loading face cascade classifier: {str(e)}")
            raise
        

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logging.error("Failed to open webcam")
        else:
            logging.info("Webcam initialized successfully")
        

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
        

        self.user_data_file = 'user_preferences.json'
        self.user_preferences = self.load_user_preferences()
        

        self.current_face_id = None
        self.face_features = {}
        self.known_faces = []
        self.permanent_users = set()
        self.showing_message = False
        self.message_time = 0
        self.message = ""
        

        self.gesture_area_height = 150
        

        self.static_gesture_count = 0
        self.last_gesture = 0
        self.gesture_cooldown = 10
        self.last_gesture_time = 0
        self.last_gesture_processed = False
        self.gesture_debug = True
        self.last_color_change_time = 0
        self.color_change_cooldown = 1.0
        

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

        permanent_preferences = {}
        for user_id in self.permanent_users:
            if user_id in self.user_preferences:
                permanent_preferences[user_id] = self.user_preferences[user_id]
        

        try:
            with open(self.user_data_file, 'w') as f:

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

                self.permanent_users.add(user_id)
                

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

        face_resized = cv2.resize(face_img, (100, 100))
        

        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        

        face_eq = cv2.equalizeHist(face_gray)
        


        gx = cv2.Sobel(face_eq, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(face_eq, cv2.CV_32F, 0, 1)
        

        mag, ang = cv2.cartToPolar(gx, gy)
        

        nbins = 16
        bins = np.linspace(0, 2*np.pi, nbins+1)
        h = np.zeros(nbins)
        for i in range(nbins):
            mask = (ang >= bins[i]) & (ang < bins[i+1])
            h[i] = np.sum(mag[mask])
        

        if np.sum(h) > 0:
            h = h / np.sum(h)
        


        cells = 4
        cell_size = face_eq.shape[0] // cells
        intensity_features = []
        
        for i in range(cells):
            for j in range(cells):
                cell = face_eq[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
                intensity_features.append(np.mean(cell))
        

        features = np.concatenate([h, np.array(intensity_features) / 255.0])
        return features
    
    def get_face_id(self, face_img):
        """
        More robust face identification based on face features
        Compares with known faces to find a match
        """

        features = self.extract_face_features(face_img)
        logging.debug("Extracted face features for identification")
        

        best_match_id = None
        best_match_score = float('inf')
        match_threshold = 0.4
        
        for known_face in self.known_faces:

            dist = np.linalg.norm(features - known_face['features'])
            

            if dist < match_threshold and dist < best_match_score:
                best_match_score = dist
                best_match_id = known_face['id']
                logging.info(f"Identified existing user with ID {best_match_id}, similarity score: {dist:.5f}")
        

        if best_match_id is None:

            new_id = f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000)}"
            

            self.face_features[new_id] = features
            

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

        height, width = frame.shape[:2]
        roi_height = height // 2
        roi = frame[height-roi_height:height, 0:width].copy()
        

        viz = roi.copy()
        

        cv2.putText(viz, "Show thumbs UP/DOWN gesture here", (width//2-180, roi_height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(viz, "Or use 'u'/'d' keys on keyboard", (width//2-150, roi_height//2+40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
        


        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        

        if not hasattr(self, 'prev_gray'):
            self.prev_gray = gray
            small_viz = cv2.resize(viz, (width//2, roi_height//2))
            frame[0:small_viz.shape[0], 0:small_viz.shape[1]] = small_viz
            return 0
        

        frame_delta = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray
        

        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        

        thresh = cv2.dilate(thresh, None, iterations=2)
        

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        

        motion_viz = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        viz_small = cv2.resize(motion_viz, (width//4, roi_height//4))
        viz[0:viz_small.shape[0], 0:viz_small.shape[1]] = viz_small
        

        motion_detected = False
        gesture_code = 0
        biggest_contour = None
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                motion_detected = True
                if area > max_area:
                    max_area = area
                    biggest_contour = contour
        

        status = "Motion Detected" if motion_detected else "No Motion"
        cv2.putText(viz, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if motion_detected else (0, 0, 255), 2)
        

        if motion_detected and biggest_contour is not None:

            cv2.drawContours(viz, [biggest_contour], -1, (0, 255, 0), 2)
            

            x, y, w, h = cv2.boundingRect(biggest_contour)
            cv2.rectangle(viz, (x, y), (x+w, y+h), (255, 0, 0), 2)
            


            roi_mid_y = roi_height // 2
            

            motion_y = y + h // 2
            

            cv2.line(viz, (0, roi_mid_y), (width, roi_mid_y), (255, 255, 0), 1)
            

            cv2.putText(viz, f"Area: {max_area}", (x, y-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            



            if motion_y < roi_mid_y:
                cv2.putText(viz, "THUMBS UP!", (width//2 - 100, roi_height-50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.rectangle(viz, (0, 0), (width, roi_height), (0, 255, 0), 5)
                print("\n*** THUMB UP DETECTED (simplified motion detection) ***\n")
                logging.info(f"Thumbs up gesture detected (area: {max_area}, position: {motion_y}/{roi_mid_y})")
                gesture_code = 1
            else:
                cv2.putText(viz, "THUMBS DOWN!", (width//2 - 120, roi_height-50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.rectangle(viz, (0, 0), (width, roi_height), (0, 0, 255), 5)
                print("\n*** THUMB DOWN DETECTED (simplified motion detection) ***\n")
                logging.info(f"Thumbs down gesture detected (area: {max_area}, position: {motion_y}/{roi_mid_y})")
                gesture_code = -1
        

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
            

            frame = cv2.flip(frame, 1)
            

            display_frame = frame.copy()
            

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            

            height, width = frame.shape[:2]
            cv2.rectangle(display_frame, (0, height-self.gesture_area_height), 
                         (width, height), (50, 50, 50), 2)
            cv2.putText(display_frame, "Gesture Area", (width//2-60, height-self.gesture_area_height+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            

            if len(faces) > 0:

                largest_face = max(faces, key=lambda rectangle: rectangle[2] * rectangle[3])
                x, y, w, h = largest_face
                

                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                

                face_img = frame[y:y+h, x:x+w]
                

                self.current_face_id = self.get_face_id(face_img)
                

                if self.current_face_id not in self.user_preferences:

                    current_features = self.face_features.get(self.current_face_id)
                    
                    self.user_preferences[self.current_face_id] = {
                        "liked_colors": [],
                        "disliked_colors": [],
                        "last_seen": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

                        "face_features": current_features.tolist() if current_features is not None else []
                    }
                    self.show_message("New user detected!")
                else:

                    self.user_preferences[self.current_face_id]["last_seen"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    

                    current_features = self.face_features.get(self.current_face_id)
                    if current_features is not None and ("face_features" not in self.user_preferences[self.current_face_id] or 
                            len(self.user_preferences[self.current_face_id]["face_features"]) == 0):
                        self.user_preferences[self.current_face_id]["face_features"] = current_features.tolist()
                

                color_name, color_rgb = self.colors[self.current_color_idx]
                

                suggestion_x = x + w + 20
                suggestion_y = y
                suggestion_width = 150
                suggestion_height = 80
                

                cv2.rectangle(display_frame, 
                             (suggestion_x-5, suggestion_y-5), 
                             (suggestion_x+suggestion_width+5, suggestion_y+suggestion_height+5), 
                             (255, 255, 255), 2)
                

                cv2.rectangle(display_frame, 
                             (suggestion_x, suggestion_y), 
                             (suggestion_x+suggestion_width, suggestion_y+suggestion_height), 
                             color_rgb, -1)
                


                cv2.putText(display_frame, color_name, (suggestion_x+3, suggestion_y+40+3), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

                cv2.putText(display_frame, color_name, (suggestion_x+3, suggestion_y+40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                

                cv2.putText(display_frame, "Thumbs Up to Like", (suggestion_x, suggestion_y+suggestion_height+20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(display_frame, "Thumbs Down to Skip", (suggestion_x, suggestion_y+suggestion_height+40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                

                user_info = f"User: {self.current_face_id[:8]}..."
                if "name" in self.user_preferences[self.current_face_id]:
                    user_info = f"User: {self.user_preferences[self.current_face_id]['name']}"
                cv2.putText(display_frame, user_info, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                

                gesture = self.detect_gesture(frame)
                

                if gesture != 0:
                    if gesture == self.last_gesture:
                        self.static_gesture_count += 1
                    else:
                        self.static_gesture_count = 1
                        self.last_gesture = gesture
                    


                    current_time = time.time()
                    enough_time_passed = current_time - self.last_gesture_time > 0.8
                    

                    if self.static_gesture_count >= 1:
                        status_color = (0, 0, 255)
                        if self.static_gesture_count >= 2:
                            status_color = (0, 255, 0)
                            

                        cv2.rectangle(display_frame, (width-150, 10), (width-10, 40), (0, 0, 0), -1)
                        cv2.putText(display_frame, f"Gesture: {self.static_gesture_count}/2",
                                    (width-140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    


                    current_time = time.time()
                    color_cooldown_passed = current_time - self.last_color_change_time > self.color_change_cooldown
                    
                    if self.static_gesture_count >= 2 and enough_time_passed and not self.last_gesture_processed and color_cooldown_passed:

                        self.last_gesture_processed = True
                        

                        prev_color_idx = self.current_color_idx
                        prev_color_name = color_name
                        

                        self.last_color_change_time = current_time
                        

                        print(f"\n=== PROCESSING GESTURE {self.last_gesture} - CHANGING COLOR ===\n")
                        

                        if gesture == 1:  

                            if color_name not in self.user_preferences[self.current_face_id]["liked_colors"]:
                                self.user_preferences[self.current_face_id]["liked_colors"].append(color_name)

                                if color_name in self.user_preferences[self.current_face_id]["disliked_colors"]:
                                    self.user_preferences[self.current_face_id]["disliked_colors"].remove(color_name)
                            

                            self.show_message(f"👍 Added {color_name} to your preferences!")
                            

                            next_color_name, _ = self.next_color()
                            

                            print(f"THUMBS UP CONFIRMED! Liked {prev_color_name}, changed to {next_color_name}")
                            

                            color_name, color_rgb = self.colors[self.current_color_idx]
                            

                            cv2.rectangle(display_frame, (10, 10), (380, 50), (0, 0, 0), -1)
                            cv2.putText(display_frame, f"COLOR CHANGED: {prev_color_name} → {next_color_name}", 
                                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            

                            self.last_gesture_time = time.time()
                            self.static_gesture_count = 0
                        

                        elif gesture == -1:  

                            if color_name not in self.user_preferences[self.current_face_id]["disliked_colors"]:
                                self.user_preferences[self.current_face_id]["disliked_colors"].append(color_name)

                                if color_name in self.user_preferences[self.current_face_id]["liked_colors"]:
                                    self.user_preferences[self.current_face_id]["liked_colors"].remove(color_name)
                            

                            self.show_message(f"👎 Skipping {color_name}, not your style!")
                            

                            next_color_name, _ = self.next_color()
                            

                            print(f"THUMBS DOWN CONFIRMED! Disliked {prev_color_name}, changed to {next_color_name}")
                            

                            color_name, color_rgb = self.colors[self.current_color_idx]
                            

                            cv2.rectangle(display_frame, (10, 10), (380, 50), (0, 0, 0), -1)
                            cv2.putText(display_frame, f"COLOR CHANGED: {prev_color_name} → {next_color_name}", 
                                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            

                            self.last_gesture_time = time.time()
                            self.static_gesture_count = 0
                else:

                    self.last_gesture_processed = False
                    

                    current_time = time.time()
                    time_since_last = current_time - self.last_gesture_time
                    if time_since_last < 1.5:
                        cooldown_percent = min(1.0, time_since_last / 1.5)
                        bar_width = int(140 * cooldown_percent)
                        

                        cv2.rectangle(display_frame, (width-150, 50), (width-10, 70), (50, 50, 50), -1)
                        cv2.rectangle(display_frame, (width-150, 50), (width-150+bar_width, 70), (0, 255, 255), -1)
                        cv2.putText(display_frame, "Cooldown", (width-140, 90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            else:
                self.current_face_id = None
            

            if self.showing_message:
                if time.time() - self.message_time < self.message_duration:

                    message_width = len(self.message) * 15
                    cv2.rectangle(display_frame, (width//2-message_width//2-10, 30), 
                                 (width//2+message_width//2+10, 70), (0, 0, 0), -1)
                    cv2.putText(display_frame, self.message, (width//2-message_width//2, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    self.showing_message = False
            

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
                        

            cv2.imshow('Face and Gesture Recognition', display_frame)
            

            key = cv2.waitKey(1) & 0xFF
            

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

                    self.user_preferences[self.current_face_id]["name"] = name
                    

                    self.permanent_users.add(self.current_face_id)
                    

                    for face in self.known_faces:
                        if face['id'] == self.current_face_id:
                            face['name'] = name
                            break
                    

                    self.save_user_preferences()
                    
                    logging.info(f"User {self.current_face_id} name set to '{name}' - marked as permanent")
                    self.show_message(f"Name set to {name}! User profile will be saved permanently.")

            elif key == ord('u'):
                if self.current_face_id and time.time() - self.last_color_change_time > self.color_change_cooldown:
                    print("\n===== KEYBOARD THUMBS UP - LIKING CURRENT COLOR =====\n")
                    color_name, _ = self.colors[self.current_color_idx]
                    

                    if "liked_colors" not in self.user_preferences[self.current_face_id]:
                        self.user_preferences[self.current_face_id]["liked_colors"] = []
                    
                    if color_name not in self.user_preferences[self.current_face_id]["liked_colors"]:
                        self.user_preferences[self.current_face_id]["liked_colors"].append(color_name)
                        logging.info(f"User {self.user_preferences[self.current_face_id]['name']} liked color: {color_name}")
                    

                    if "disliked_colors" in self.user_preferences[self.current_face_id] and \
                       color_name in self.user_preferences[self.current_face_id]["disliked_colors"]:
                        self.user_preferences[self.current_face_id]["disliked_colors"].remove(color_name)
                    

                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (width, height), (0, 255, 0), -1)
                    cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
                    

                    next_color_name, _ = self.next_color()
                    self.show_message(f"👍 LIKED: {color_name} → Now showing: {next_color_name}")
                    self.last_color_change_time = time.time()
            

            elif key == ord('d'):
                if self.current_face_id and time.time() - self.last_color_change_time > self.color_change_cooldown:
                    print("\n===== KEYBOARD THUMBS DOWN - DISLIKING CURRENT COLOR =====\n")
                    color_name, _ = self.colors[self.current_color_idx]
                    

                    if "disliked_colors" not in self.user_preferences[self.current_face_id]:
                        self.user_preferences[self.current_face_id]["disliked_colors"] = []
                    
                    if color_name not in self.user_preferences[self.current_face_id]["disliked_colors"]:
                        self.user_preferences[self.current_face_id]["disliked_colors"].append(color_name)
                        logging.info(f"User {self.user_preferences[self.current_face_id]['name']} disliked color: {color_name}")
                    

                    if "liked_colors" in self.user_preferences[self.current_face_id] and \
                       color_name in self.user_preferences[self.current_face_id]["liked_colors"]:
                        self.user_preferences[self.current_face_id]["liked_colors"].remove(color_name)
                    

                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
                    

                    next_color_name, _ = self.next_color()
                    self.show_message(f"👎 DISLIKED: {color_name} → Now showing: {next_color_name}")
                    self.last_color_change_time = time.time()
        

        self.save_user_preferences()
        logging.info("User preferences saved")
        self.cap.release()
        cv2.destroyAllWindows()
        logging.info("Application closed")
        print("Application closed.")

if __name__ == "__main__":

    system = FaceGestureColorSystem()
    system.run()
