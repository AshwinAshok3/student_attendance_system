import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Mute the spammy warnings
warnings.filterwarnings("ignore")

# PATHS
MODEL_ROOT = os.path.abspath("./ai_models")
DATA_ROOT = os.path.abspath("./data/students")
DB_PATH = "./database/embeddings.pkl"
SIMILARITY_THRESHOLD = 0.5

class SentinelEngine:
    def __init__(self):
        print("â³ [AI] Initializing InsightFace (GPU)...")
        self.app = FaceAnalysis(name='buffalo_l', root=MODEL_ROOT, providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        self.known_embeddings = []
        self.known_names = []
        
        # 1. Load existing DB
        self.load_embeddings()
        
        # 2. SYNC with Data Folder (The New Feature)
        self.sync_data_folder()
        print("âœ… [AI] System Ready.")

    def load_embeddings(self):
        if os.path.exists(DB_PATH):
            try:
                with open(DB_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.known_embeddings = data['embeddings']
                    self.known_names = data['names']
            except:
                self.known_embeddings = []
                self.known_names = []

    def save_embeddings(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        with open(DB_PATH, 'wb') as f:
            pickle.dump({'embeddings': self.known_embeddings, 'names': self.known_names}, f)

    def sync_data_folder(self):
        """
        Scans /data/students folder.
        If a folder name exists but isn't in the DB, it learns the faces inside.
        """
        if not os.path.exists(DATA_ROOT):
            os.makedirs(DATA_ROOT, exist_ok=True)
            return

        print(f"ðŸ”„ [SYNC] Scanning {DATA_ROOT} for new faces...")
        new_faces_added = 0
        
        # Loop through every student folder
        for student_name in os.listdir(DATA_ROOT):
            student_path = os.path.join(DATA_ROOT, student_name)
            
            if not os.path.isdir(student_path):
                continue
                
            # Check if we already know this person
            if student_name in self.known_names:
                continue
                
            print(f"   ðŸ‘‰ Learning new student: {student_name}")
            
            # Process images in that folder
            valid_embedding = None
            for img_file in os.listdir(student_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(student_path, img_file)
                    img = cv2.imread(img_path)
                    if img is None: continue
                    
                    faces = self.app.get(img)
                    if len(faces) == 1:
                        valid_embedding = faces[0].embedding
                        break # Found a good face, stop scanning this folder
            
            if valid_embedding is not None:
                self.known_embeddings.append(valid_embedding)
                self.known_names.append(student_name)
                new_faces_added += 1
            else:
                print(f"   âš ï¸ Could not find a clear face for {student_name}")

        if new_faces_added > 0:
            self.save_embeddings()
            print(f"âœ… [SYNC] Added {new_faces_added} new students from files.")
        else:
            print("âœ… [SYNC] Database is up to date.")

    def process_frame(self, frame):
        faces = self.app.get(frame)
        results = []
        for face in faces:
            name = "Unknown"
            status = "Unregistered"
            color = (0, 0, 255)
            
            if len(self.known_embeddings) > 0:
                sims = cosine_similarity([face.embedding], self.known_embeddings)[0]
                best_idx = np.argmax(sims)
                if sims[best_idx] > SIMILARITY_THRESHOLD:
                    name = self.known_names[best_idx]
                    status = "Recognized"
                    color = (0, 255, 0)
            
            results.append({
                "name": name,
                "status": status,
                "bbox": face.bbox.astype(int),
                "color": color
            })
        return results

    def register_new_face(self, frame, new_name):
        faces = self.app.get(frame)
        if len(faces) == 0: return False, "âŒ No face detected."
        
        # Check strict "One Unknown" rule
        unknown_count = 0
        target_face = None
        
        for face in faces:
            is_known = False
            if len(self.known_embeddings) > 0:
                sims = cosine_similarity([face.embedding], self.known_embeddings)[0]
                if np.max(sims) > SIMILARITY_THRESHOLD: is_known = True
            
            if not is_known:
                unknown_count += 1
                target_face = face

        if unknown_count == 0: return False, "âš ï¸ Face already registered."
        if unknown_count > 1: return False, "âŒ Multiple unknown people detected."

        self.known_embeddings.append(target_face.embedding)
        self.known_names.append(new_name)
        self.save_embeddings()
        
        save_dir = os.path.join(DATA_ROOT, new_name)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, "ref_img.jpg"), frame)
        
        return True, f"âœ… Registered {new_name}"
