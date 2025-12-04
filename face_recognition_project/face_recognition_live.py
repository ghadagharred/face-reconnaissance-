import cv2
import face_recognition
import os
import pickle
from datetime import datetime
from picamera2 import Picamera2

KNOWN_FACES_DIR = "dataset"
UNKNOWN_FACES_DIR = "unknown_faces"
os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)

ENCODINGS_FILE = "face_encodings.pkl"

if not os.path.exists(ENCODINGS_FILE):
    print("Creation des encodages des visages connus...")
    encodings = []
    names = []

    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            image = face_recognition.load_image_file(img_path)
            face_encs = face_recognition.face_encodings(image)
            if face_encs:
                encodings.append(face_encs[0])
                names.append(person_name)

    data = {"encodings": encodings, "names": names}
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)
    print("Encodages enregistres dans", ENCODINGS_FILE)
else:
    print("Encodages existants charges.")
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)

# === Initialiser la camera ===
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

print("Camera OK, reconnaissance faciale demarree")

# === Boucle principale ===
while True:
    frame = picam2.capture_array()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(data["encodings"], face_encoding)
        name = "Inconnu"

        if True in matches:
            idx = matches.index(True)
            name = data["names"][idx]
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unknown_path = os.path.join(UNKNOWN_FACES_DIR, f"unknown_{timestamp}.jpg")
            cv2.imwrite(unknown_path, frame)
            print(f"Visage inconnu detecte, sauvegarde dans {unknown_path}")

        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Reconnaissance faciale", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC pour quitter
        break

picam2.stop()
cv2.destroyAllWindows()
