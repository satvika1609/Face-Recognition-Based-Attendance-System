import cv2
import dlib
import os
import time
import logging

FACE_STORAGE_PATH = "data/data_faces_from_camera"
os.makedirs(FACE_STORAGE_PATH, exist_ok=True)

detector = dlib.get_frontal_face_detector()

def get_next_person_id():
    people = [
        d for d in os.listdir(FACE_STORAGE_PATH)
        if d.startswith("person_") and d.split("_")[1].isdigit()
    ]
    ids = [int(p.split("_")[1]) for p in people]
    return max(ids, default=0) + 1

def capture_faces(person_name, image_count=5):
    person_id = get_next_person_id()
    folder_name = f"person_{person_id}_{person_name}"
    save_path = os.path.join(FACE_STORAGE_PATH, folder_name)
    os.makedirs(save_path, exist_ok=True)

    print(f"[INFO] Saving face images to: {save_path}")

    cap = cv2.VideoCapture(0)
    saved_count = 0

    print("[INFO] Face capture started. Press 'q' to cancel.")

    while cap.isOpened() and saved_count < image_count:
        success, frame = cap.read()
        if not success:
            print("[ERROR] Failed to access camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cropped_face = frame[y1:y2, x1:x2]
            if cropped_face.size > 0:
                image_path = os.path.join(save_path, f"img_face_{saved_count + 1}.jpg")
                cv2.imwrite(image_path, cropped_face)
                print(f"[INFO] Saved image {saved_count + 1}: {image_path}")
                saved_count += 1
                time.sleep(0.5)

        cv2.imshow("Registering Face - Press 'q' to exit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if saved_count == image_count:
        print(f"[INFO] Face registration completed for '{person_name}'")
    else:
        print(f"[INFO] Registration incomplete. Only {saved_count} images saved.")

def run():
    logging.basicConfig(level=logging.INFO)
    name = input("Enter the name of the person to register: ").strip()
    if name:
        capture_faces(name)
    else:
        print("[WARNING] Name cannot be empty.")

if __name__ == '__main__':
    run()
