# tools/homography_calibrator.py
import json
from pathlib import Path

import cv2
import numpy as np


class HomographyCalibrator:
    def __init__(self):
        self.points_image = []
        self.frame = None
        self.clone = None

    def _redraw(self):
        self.clone = self.frame.copy()

        for i, point in enumerate(self.points_image):
            x, y = point

            cv2.circle(self.clone, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(
                self.clone,
                str(i + 1),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        if len(self.points_image) > 1:
            cv2.polylines(
                self.clone,
                [np.array(self.points_image, dtype=np.int32)],
                False,
                (0, 255, 0),
                1,
            )

        if len(self.points_image) == 4:
            cv2.polylines(
                self.clone,
                [np.array(self.points_image, dtype=np.int32)],
                True,
                (0, 255, 0),
                1,
            )

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points_image) < 4:
            self.points_image.append([x, y])
            self._redraw()
            print(f"Point {len(self.points_image)} : ({x}, {y})")

    def pick_points(self, frame):
        self.frame = frame
        self.clone = frame.copy()
        self.points_image = []

        cv2.namedWindow("calibration")
        cv2.setMouseCallback("calibration", self._mouse_callback)

        print("Clique 4 points dans l'ordre :")
        print("1. haut-gauche")
        print("2. haut-droit")
        print("3. bas-droit")
        print("4. bas-gauche")
        print("'u' : annuler le dernier point")
        print("'r' : réinitialiser")
        print("ENTRÉE : valider")

        while True:
            cv2.imshow("calibration", self.clone)
            key = cv2.waitKey(1) & 0xFF

            if key == 13 and len(self.points_image) == 4:
                break

            if key == ord("u") and self.points_image:
                self.points_image.pop()
                self._redraw()

            if key == ord("r"):
                self.points_image = []
                self._redraw()

        cv2.destroyAllWindows()

    def enter_world_coords(self):
        points_world = []

        print("\nEntre les coordonnées réelles en mètres pour chaque point.")

        for i in range(4):
            x = float(input(f"Point {i + 1} - X (m) : "))
            y = float(input(f"Point {i + 1} - Y (m) : "))
            points_world.append([x, y])

        return points_world

    def compute_homography(self, points_world):
        src = np.array(self.points_image, dtype=np.float32)
        dst = np.array(points_world, dtype=np.float32)

        H, _ = cv2.findHomography(src, dst)

        if H is None:
            raise RuntimeError("Impossible de calculer l'homographie.")

        return H

    def save(self, H, points_world, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "points_image": self.points_image,
            "points_world": points_world,
            "homography": H.tolist(),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Homographie sauvegardée : {output_path}")
