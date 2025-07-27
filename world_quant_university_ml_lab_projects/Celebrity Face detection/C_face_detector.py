# Import libraries
import os, shutil, torch, cv2
from pathlib import Path
from PIL import Image
from torchvision import datasets
from torch.utils.data import DataLoader
from facenet_pytorch import MTCNN, InceptionResnetV1

# Set device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Extract frames from video
frames_dir = Path('data/extracted_frames')
frames_dir.mkdir(exist_ok=True)
video_capture = cv2.VideoCapture('data/lupita_nyongo.mp4')
interval, frame_count = 6, 0
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    if frame_count % interval == 0:
        cv2.imwrite(str(frames_dir / f"frame_{frame_count}.jpg"), frame)
    frame_count += 1
video_capture.release()

# Organize known faces (Lupita & Christoph) into folders
images_dir = Path('data/images')
(images_dir / 'lupita').mkdir(exist_ok=True)
(images_dir / 'christoph').mkdir(exist_ok=True)
lupita_imgs = ["frame_3438.jpg", "frame_3486.jpg", "frame_3852.jpg", "frame_4062.jpg", "frame_4914.jpg", "frame_4866.jpg"]
christoph_imgs = ["frame_54.jpg", "frame_66.jpg", "frame_72.jpg", "frame_108.jpg", "frame_186.jpg", "frame_246.jpg"]

for img in lupita_imgs:
    shutil.copy(frames_dir / img, images_dir / 'lupita')
for img in christoph_imgs:
    shutil.copy(frames_dir / img, images_dir / 'christoph')

# Initialize face detector (MTCNN) and feature extractor (ResNet)
mtcnn = MTCNN(keep_all=True, min_face_size=40)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Extract face embeddings for known people
dataset = datasets.ImageFolder(images_dir)
idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
name_to_embeddings = {name: [] for name in idx_to_class.values()}

for img, idx in DataLoader(dataset, collate_fn=lambda x: x[0]):
    face, prob = mtcnn(img, return_prob=True)
    if face is not None and prob >= 0.90:
        emb = resnet(face[0].unsqueeze(0))
        name_to_embeddings[idx_to_class[idx]].append(emb)

# Compute average face embeddings
avg_embedding_lupita = torch.mean(torch.stack(name_to_embeddings["lupita"]), dim=0)
avg_embedding_christoph = torch.mean(torch.stack(name_to_embeddings["christoph"]), dim=0)

# Recognize faces from test frames
test_paths = [frames_dir / frame for frame in ["frame_2658.jpg", "frame_4614.jpg", "frame_972.jpg", "frame_30.jpg"]]
from utils import recognize_faces  # Prewritten function in utils.py
embedding_data = [(avg_embedding_lupita, "lupita"), (avg_embedding_christoph, "christoph")]

recognized_faces = []
for test_img_path in test_paths:
    recognized_faces.append(recognize_faces(test_img_path, embedding_data, mtcnn, resnet))
