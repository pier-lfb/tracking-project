import cv2
import numpy as np


class TeamAssigner:
    """Assigne une équipe à partir de la couleur dominante du maillot."""

    def __init__(self, min_samples=150, max_samples=400, num_teams=2, min_margin=12.0):
        self.min_samples = int(min_samples)
        self.max_samples = int(max_samples)
        self.num_teams = int(num_teams)
        self.min_margin = float(min_margin)
        self.samples = []
        self.centers = None

    @property
    def is_fitted(self):
        return self.centers is not None

    def collect(self, frame, player_boxes):
        """Accumule des couleurs de maillot puis fit le clustering."""
        if self.is_fitted:
            return True

        for box in player_boxes:
            color = self.extract_jersey_color(frame, box)
            if color is not None:
                self.samples.append(color)

        self.samples = self.samples[-self.max_samples:]

        if len(self.samples) < self.min_samples:
            return False

        self._fit()
        return True

    def predict(self, frame, box):
        if not self.is_fitted:
            return -1

        color = self.extract_jersey_color(frame, box)
        if color is None:
            return -1

        distances = np.linalg.norm(self.centers - color, axis=1)
        order = np.argsort(distances)

        best_team = int(order[0])
        best_dist = distances[order[0]]
        second_dist = distances[order[1]]

        # Si les deux équipes sont trop proches, on refuse de voter
        if second_dist - best_dist < self.min_margin:
            return -1

        return best_team

    def extract_jersey_color(self, frame, box):
        """Extrait la couleur médiane du maillot en LAB."""
        jersey = self._crop_jersey(frame, box)
        if jersey is None:
            return None

        jersey = cv2.resize(jersey, (12, 12), interpolation=cv2.INTER_AREA)
        grass_mask = self._grass_mask(jersey)

        lab = cv2.cvtColor(jersey, cv2.COLOR_BGR2LAB)
        pixels = lab.reshape(-1, 3).astype(np.float32)
        valid_pixels = pixels[~grass_mask]

        if len(valid_pixels) < 5:
            valid_pixels = pixels

        return np.median(valid_pixels, axis=0)

    def _fit(self):
        data = np.asarray(self.samples, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            50,
            0.5,
        )
        _, _, centers = cv2.kmeans(
            data,
            self.num_teams,
            None,
            criteria,
            10,
            cv2.KMEANS_PP_CENTERS,
        )
        self.centers = centers
        self.samples = []

    @staticmethod
    def _crop_jersey(frame, box):
        x1, y1, x2, y2 = np.asarray(box).astype(int)
        image_h, image_w = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_w, x2)
        y2 = min(image_h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        h, w = crop.shape[:2]

        if w < 6 or h < 10:
            return None

        # Haut-centre du joueur : on évite short, jambes et pelouse.
        jersey = crop[int(0.18 * h):int(0.58 * h), int(0.20 * w):int(0.80 * w)]
        if jersey.size == 0:
            return None
        return jersey

    @staticmethod
    def _grass_mask(bgr_image):
        pixels = bgr_image.reshape(-1, 3).astype(np.float32)
        b = pixels[:, 0]
        g = pixels[:, 1]
        r = pixels[:, 2]
        return (g > 1.15 * r) & (g > 1.15 * b) & (g > 45)
