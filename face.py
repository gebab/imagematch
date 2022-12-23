import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np

class Face:
    def __init__(self):
        self.mtcnn = MTCNN(keep_all=True)

        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

    def embedding(self,img_path):
        image = Image.open(img_path)
        boxes, _ = self.mtcnn.detect(image)
        if boxes is not None:
            x, y, width, height = boxes[0]
            face = image.crop((x, y, x+width, y+height))
            face_resize = face.resize((160, 160))
            face_array = np.array(face_resize)
            face_tensor = torch.tensor(face_array, dtype=torch.uint8).float()
            face_tensor = face_tensor.unsqueeze(0)
            face_tensor = face_tensor.permute(0, 3, 1, 2)
            embedding = self.resnet(face_tensor).detach().numpy()
            return [True, face, embedding[0]]
        return [False, -1, -1]

