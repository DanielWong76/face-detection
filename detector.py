from pathlib import Path
import pickle
import face_recognition
from collections import Counter
from PIL import Image, ImageDraw
import cv2
from datetime import datetime

BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

def recognize_faces(
    image_location: str,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)
    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    results = []
    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        
        results.append((bounding_box, name))
        # Removed print(name, bounding_box)
        _display_face(draw, bounding_box, name)

    del draw
    pillow_image.show()
    return results
	
def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill="blue",
        outline="blue",
    )
    draw.text(
        (text_left, text_top),
        name,
        fill="white",
    )

def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]

def recognize_faces_from_webcam(
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    video_capture = cv2.VideoCapture(0)
    recognized_people = set()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB

        face_locations = face_recognition.face_locations(rgb_frame, model=model)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for bounding_box, unknown_encoding in zip(face_locations, face_encodings):
            name = _recognize_face(unknown_encoding, loaded_encodings)
            if not name:
                name = "Unknown"
            _display_face_opencv(frame, bounding_box, name)

            if name != "Unknown" and name not in recognized_people:
                _zoom_and_save(frame, bounding_box, name)
                recognized_people.add(name)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def _zoom_and_save(frame, bounding_box, name):
    top, right, bottom, left = bounding_box

    # Define the zoom factor
    zoom_factor = 1.5

    # Compute the new bounding box coordinates
    center_x, center_y = (left + right) // 2, (top + bottom) // 2
    width = int((right - left) * zoom_factor)
    height = int((bottom - top) * zoom_factor)
    new_left = max(center_x - width // 2, 0)
    new_right = min(center_x + width // 2, frame.shape[1])
    new_top = max(center_y - height // 2, 0)
    new_bottom = min(center_y + height // 2, frame.shape[0])

    # Crop the zoomed-in region
    zoomed_frame = frame[new_top:new_bottom, new_left:new_right]

    # Save the image
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"recognized_faces/{name}_{timestamp}.jpg"
    Path("recognized_faces").mkdir(exist_ok=True)
    cv2.imwrite(filename, zoomed_frame)

def _display_face_opencv(frame, bounding_box, name):
    top, right, bottom, left = bounding_box
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

#recognize_faces_from_webcam()