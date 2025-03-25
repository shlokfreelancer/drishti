import cv2

class Camera:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)

    def get_frame(self):
        """Captures a single frame from the camera."""
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def release(self):
        """Releases the camera resource."""
        self.cap.release()
        cv2.destroyAllWindows()

# Testing the camera
if __name__ == "__main__":
    cam = Camera()
    while True:
        frame = cam.get_frame()
        if frame is not None:
            cv2.imshow("Drishti Smart Glasses - Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Press 'q' to quit

    cam.release()
