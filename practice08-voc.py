"""
Image Search using Bag of Words (BoW) and Vocabulary Tree
- Loads images from resource/sfm_images
- Builds vocabulary tree from image features
- Adds images to BoW database
- Queries a test image

Cache behavior (required):
- Saves vocabulary tree + BoW DB to sft_output/
- If cache files exist, loads them and skips rebuilding
"""

from __future__ import annotations

import os
import glob
import pickle
from dataclasses import dataclass
import argparse

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans


CACHE_DIR = "resource/voc_output"
VOCAB_CACHE_NAME = "voctree.pkl"
BOW_CACHE_NAME = "bow_db.pkl"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _list_images(image_folder: str) -> list[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths: list[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(image_folder, ext)))
    paths = sorted(paths)
    return paths


def _as_float32(descriptors: np.ndarray | None) -> np.ndarray | None:
    if descriptors is None:
        return None
    if descriptors.dtype != np.float32:
        return descriptors.astype(np.float32)
    return descriptors


class VocabularyTree:
    """A practical (flat) vocabulary quantizer used as a "vocabulary tree" stand-in.

    For class exercises, this is usually sufficient: build a visual vocabulary with k-means,
    then quantize descriptors into visual word IDs.
    """

    def __init__(self, k: int = 10, levels: int = 3):
        self.k = int(k)
        self.levels = int(levels)
        self.vocabulary_size = self.k ** self.levels
        self.kmeans: MiniBatchKMeans | None = None

    def build(self, descriptors: np.ndarray) -> None:
        descriptors = _as_float32(descriptors)
        if descriptors is None or len(descriptors) == 0:
            raise ValueError("No descriptors provided to build vocabulary")

        target_clusters = min(self.vocabulary_size, len(descriptors))
        print(f"Building vocabulary (k={self.k}, levels={self.levels})")
        print(f"- target clusters: {target_clusters}")
        print(f"- total descriptors: {len(descriptors)}")

        self.kmeans = MiniBatchKMeans(
            n_clusters=target_clusters,
            batch_size=1000,
            random_state=42,
            verbose=0,
        )
        self.kmeans.fit(descriptors)
        self.vocabulary_size = int(self.kmeans.n_clusters)
        print(f"Vocabulary built with {self.vocabulary_size} visual words")

    def quantize(self, descriptors: np.ndarray | None) -> np.ndarray:
        if descriptors is None or len(descriptors) == 0:
            return np.array([], dtype=np.int32)
        if self.kmeans is None:
            raise RuntimeError("VocabularyTree is not built/loaded")
        descriptors = _as_float32(descriptors)
        return self.kmeans.predict(descriptors).astype(np.int32)

    def save(self, filepath: str) -> None:
        if self.kmeans is None:
            raise RuntimeError("Cannot save: vocabulary is not built")
        _ensure_dir(os.path.dirname(filepath) or ".")
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "k": self.k,
                    "levels": self.levels,
                    "kmeans": self.kmeans,
                    "vocabulary_size": self.vocabulary_size,
                },
                f,
            )

    @classmethod
    def load(cls, filepath: str) -> "VocabularyTree":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        vt = cls(k=int(data["k"]), levels=int(data["levels"]))
        vt.kmeans = data["kmeans"]
        vt.vocabulary_size = int(data.get("vocabulary_size", vt.kmeans.n_clusters))
        return vt


class BagOfWords:
    def __init__(self, vocabulary_tree: VocabularyTree):
        self.vocab_tree = vocabulary_tree
        self.bow_vectors: dict[str, np.ndarray] = {}
        self.idf_weights: np.ndarray | None = None
        self.image_paths: list[str] = []

    def compute_bow(self, descriptors: np.ndarray | None) -> np.ndarray:
        vocab_size = self.vocab_tree.vocabulary_size
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(vocab_size, dtype=np.float32)

        visual_words = self.vocab_tree.quantize(descriptors)
        bow = np.bincount(visual_words, minlength=vocab_size).astype(np.float32)
        s = float(bow.sum())
        if s > 0:
            bow /= s
        return bow

    def add_image(self, image_path: str, descriptors: np.ndarray | None) -> None:
        bow = self.compute_bow(descriptors)
        # Diagnostic: check BoW vector
        nnz = int(np.sum(bow > 0))
        if len(self.bow_vectors) < 3:  # Show first few
            print(f"  BoW for {os.path.basename(image_path)}: {nnz} non-zero bins, sum={bow.sum():.4f}")
        self.bow_vectors[image_path] = bow
        self.image_paths.append(image_path)

    def compute_tf_idf(self) -> None:
        if not self.bow_vectors:
            raise RuntimeError("No images in BoW database")

        n_images = len(self.bow_vectors)
        vocab_size = self.vocab_tree.vocabulary_size

        # Compute document frequency
        df = np.zeros(vocab_size, dtype=np.float32)
        for v in self.bow_vectors.values():
            df += (v > 0).astype(np.float32)

        # IDF with smoothing (add 1 to prevent log(0) and extreme weights)
        self.idf_weights = np.log((n_images + 1.0) / (df + 1.0)) + 1.0
        self.idf_weights = self.idf_weights.astype(np.float32)

        # Diagnostic
        nonzero_words = int(np.sum(df > 0))
        print(f"TF-IDF: {nonzero_words}/{vocab_size} visual words used ({nonzero_words/vocab_size*100:.1f}%)")
        print(f"  IDF range: [{self.idf_weights.min():.3f}, {self.idf_weights.max():.3f}]")
        
        # Apply TF-IDF and L2 normalize
        zero_count = 0
        for key, v in list(self.bow_vectors.items()):
            # TF-IDF weighting
            w = v * self.idf_weights
            
            # L2 normalize
            norm = float(np.linalg.norm(w))
            if norm > 1e-10:
                w /= norm
            else:
                # Fallback: if zero, just keep raw BoW (shouldn't happen with proper descriptors)
                zero_count += 1
                print(f"    WARNING: {os.path.basename(key)} has zero norm, keeping raw BoW")
            
            self.bow_vectors[key] = w
        
        print(f"  TF-IDF complete: {n_images - zero_count}/{n_images} images OK")

    def query(self, query_descriptors: np.ndarray | None, top_k: int = 5) -> list[tuple[str, float]]:
        q = self.compute_bow(query_descriptors)

        if self.idf_weights is not None:
            q = (q * self.idf_weights).astype(np.float32)
            norm = float(np.linalg.norm(q))
            if norm > 1e-10:
                q /= norm

        scores: list[tuple[str, float]] = []
        for path, v in self.bow_vectors.items():
            scores.append((path, float(np.dot(q, v))))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def save(self, filepath: str) -> None:
        _ensure_dir(os.path.dirname(filepath) or ".")
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "image_paths": self.image_paths,
                    "bow_vectors": self.bow_vectors,
                    "idf_weights": self.idf_weights,
                },
                f,
            )

    @classmethod
    def load(cls, filepath: str, vocabulary_tree: VocabularyTree) -> "BagOfWords":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        bow = cls(vocabulary_tree)
        bow.image_paths = data["image_paths"]
        bow.bow_vectors = data["bow_vectors"]
        bow.idf_weights = data["idf_weights"]
        return bow


class ImageSearchEngine:
    def __init__(self, k: int = 100, levels: int = 2):
        self.k = int(k)
        self.levels = int(levels)
        self.vocab_tree: VocabularyTree | None = None
        self.bow: BagOfWords | None = None

        # Prefer SIFT; fall back to ORB if SIFT not available.
        try:
            self.feature_detector = cv2.SIFT_create()
            self._descriptor_is_binary = False
            print("Using SIFT features")
        except Exception:
            self.feature_detector = cv2.ORB_create(nfeatures=2000)
            self._descriptor_is_binary = True
            print("SIFT not available; using ORB features")

    def extract_features(self, image_path: str):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None
        keypoints, descriptors = self.feature_detector.detectAndCompute(img, None)
        # ORB descriptors are uint8; convert to float for k-means
        descriptors = _as_float32(descriptors)
        return keypoints, descriptors


    def extract_features_from_gray(self, img_gray: np.ndarray):
        keypoints, descriptors = self.feature_detector.detectAndCompute(img_gray, None)
        descriptors = _as_float32(descriptors)
        return keypoints, descriptors

    def search_frame(self, frame_bgr: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        if self.bow is None:
            raise RuntimeError("Database not built/loaded")

        img_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        _, desc = self.extract_features_from_gray(img_gray)
        if desc is None or len(desc) == 0:
            return []
        return self.bow.query(desc, top_k=top_k)

    def add_frame_to_database(self, frame_bgr: np.ndarray, image_path: str) -> bool:
        '''Adds the current frame to the BoW database using the existing vocabulary.

        Notes:
        - Uses the current (precomputed) IDF weights if present.
        - Does NOT recompute IDF globally (keeps it fast for live usage).
        '''
        if self.bow is None:
            raise RuntimeError("Database not built/loaded")

        img_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        _, desc = self.extract_features_from_gray(img_gray)
        if desc is None or len(desc) == 0:
            return False

        bow = self.bow.compute_bow(desc)
        if self.bow.idf_weights is not None:
            w = (bow * self.bow.idf_weights).astype(np.float32)
            norm = float(np.linalg.norm(w))
            if norm > 0:
                w /= norm
            self.bow.bow_vectors[image_path] = w
        else:
            self.bow.bow_vectors[image_path] = bow.astype(np.float32)

        self.bow.image_paths.append(image_path)
        return True

    def _cache_paths(self) -> tuple[str, str]:
        _ensure_dir(CACHE_DIR)
        voc_path = os.path.join(CACHE_DIR, VOCAB_CACHE_NAME)
        bow_path = os.path.join(CACHE_DIR, BOW_CACHE_NAME)
        return voc_path, bow_path

    def try_load_cache(self) -> bool:
        voc_path, bow_path = self._cache_paths()
        if not (os.path.exists(voc_path) and os.path.exists(bow_path)):
            return False

        print(f"Loading cached vocabulary tree: {voc_path}")
        self.vocab_tree = VocabularyTree.load(voc_path)

        print(f"Loading cached BoW database: {bow_path}")
        self.bow = BagOfWords.load(bow_path, self.vocab_tree)

        return True

    def save_cache(self) -> None:
        if self.vocab_tree is None or self.bow is None:
            raise RuntimeError("Nothing to save")
        voc_path, bow_path = self._cache_paths()
        self.vocab_tree.save(voc_path)
        self.bow.save(bow_path)
        print(f"Saved cache to {CACHE_DIR}/ ({VOCAB_CACHE_NAME}, {BOW_CACHE_NAME})")

    def build_database(self, image_folder: str, force_rebuild: bool = False) -> None:
        if not force_rebuild and self.try_load_cache():
            print("Cache found; skipping rebuild")
            return

        image_paths = _list_images(image_folder)
        if not image_paths:
            raise FileNotFoundError(f"No images found in {image_folder}")

        print(f"Found {len(image_paths)} images in {image_folder}")

        all_desc: list[np.ndarray] = []
        per_image_desc: dict[str, np.ndarray] = {}

        print("Extracting features...")
        for i, img_path in enumerate(image_paths, start=1):
            _, desc = self.extract_features(img_path)
            if desc is None or len(desc) == 0:
                continue
            per_image_desc[img_path] = desc
            all_desc.append(desc)
            if i % 5 == 0 or i == len(image_paths):
                print(f"- {i}/{len(image_paths)}")

        if not all_desc:
            raise RuntimeError("No descriptors extracted from any image")

        all_desc_mat = np.vstack(all_desc)

        self.vocab_tree = VocabularyTree(k=self.k, levels=self.levels)
        self.vocab_tree.build(all_desc_mat)

        self.bow = BagOfWords(self.vocab_tree)
        for img_path, desc in per_image_desc.items():
            self.bow.add_image(img_path, desc)

        self.bow.compute_tf_idf()

        self.save_cache()

    def search(self, query_image_path: str, top_k: int = 5) -> list[tuple[str, float]]:
        if self.bow is None:
            raise RuntimeError("Database not built/loaded")

        _, desc = self.extract_features(query_image_path)
        if desc is None or len(desc) == 0:
            print("No features in query image")
            return []

        return self.bow.query(desc, top_k=top_k)

    @staticmethod
    def visualize_results(query_image_path: str, results: list[tuple[str, float]], max_display: int = 5) -> None:
        """Visualize query + top retrieved images using matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("matplotlib not available; skipping visualization (pip install matplotlib)")
            return

        n_results = min(len(results), max_display)
        n_cols = 1 + n_results

        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        if n_cols == 1:
            axes = [axes]

        def _load_rgb(p: str):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                return None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        qimg = _load_rgb(query_image_path)
        axes[0].imshow(qimg if qimg is not None else np.zeros((10, 10, 3), dtype=np.uint8))
        axes[0].set_title(f"Query\n{os.path.basename(query_image_path)}")
        axes[0].axis("off")

        for i, (img_path, score) in enumerate(results[:n_results], start=1):
            rimg = _load_rgb(img_path)
            axes[i].imshow(rimg if rimg is not None else np.zeros((10, 10, 3), dtype=np.uint8))
            axes[i].set_title(f"#{i}\n{os.path.basename(img_path)}\n{score:.3f}")
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()




def _resize_to_fit(img_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    if img_bgr is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    scale = min(width / w, height / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y0 = (height - new_h) // 2
    x0 = (width - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _draw_thumbnail_column(
    image_paths: list[str],
    title: str,
    width: int,
    height: int,
    slots: int,
    highlight_first: bool = False,
) -> np.ndarray:
    col = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(col, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    if slots <= 0:
        return col

    thumb_h = max(1, (height - 40) // slots)
    y = 40

    for i, p in enumerate(image_paths[:slots]):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        thumb = _resize_to_fit(img, width, thumb_h)
        col[y : y + thumb_h, 0:width] = thumb

        label = os.path.basename(p)
        cv2.rectangle(col, (0, y), (width - 1, y + thumb_h - 1), (50, 50, 50), 1)
        cv2.putText(col, label[:24], (8, y + thumb_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        if highlight_first and i == 0:
            cv2.rectangle(col, (2, y + 2), (width - 3, y + thumb_h - 3), (0, 255, 0), 2)

        y += thumb_h
        if y >= height:
            break

    return col


def _draw_results_column(
    results: list[tuple[str, float]],
    title: str,
    width: int,
    height: int,
    slots: int = 5,
    highlight_first: bool = True,
) -> np.ndarray:
    col = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(col, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    if slots <= 0:
        return col

    thumb_h = max(1, (height - 40) // slots)
    y = 40

    for rank, (img_path, score) in enumerate(results[:slots], start=1):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        thumb = _resize_to_fit(img, width, thumb_h)
        col[y : y + thumb_h, 0:width] = thumb

        cv2.rectangle(col, (0, y), (width - 1, y + thumb_h - 1), (50, 50, 50), 1)

        base = os.path.basename(img_path)
        line1 = f"#{rank}  {base[:18]}"
        line2 = f"score={score:.3f}"
        cv2.putText(col, line1, (8, y + thumb_h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(col, line2, (8, y + thumb_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        if highlight_first and rank == 1:
            cv2.rectangle(col, (2, y + 2), (width - 3, y + thumb_h - 3), (0, 255, 0), 2)

        y += thumb_h
        if y >= height:
            break

    return col



def run_camera_stream(
    engine: 'ImageSearchEngine',
    top_k: int = 5,
    camera_index: int = 0,
    pane_h: int = 480,
    left_w: int = 320,
    mid_w: int = 640,
    right_w: int = 320,
) -> None:
    if engine.bow is None:
        raise RuntimeError('Database not built/loaded')

    top_k = int(min(max(1, top_k), 5))

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open camera index {camera_index}')

    _ensure_dir(os.path.join(CACHE_DIR, 'captured'))

    added_paths: list[str] = []

    win = 'BoW Image Search (a=add, q/esc=quit)'

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            results = engine.search_frame(frame, top_k=top_k)

            left = _draw_thumbnail_column(
                list(reversed(added_paths)),
                title=f'Added ({len(added_paths)})',
                width=left_w,
                height=pane_h,
                slots=6,
            )

            mid = _resize_to_fit(frame, mid_w, pane_h)
            cv2.putText(mid, 'Camera', (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            right = _draw_results_column(
                results,
                title=f'Top-{min(top_k, len(results))} Nearest',
                width=right_w,
                height=pane_h,
                slots=5,
                highlight_first=True,
            )

            # Robust composition (avoids np.hstack shape surprises)
            def _force_bgr(img: np.ndarray, width: int, height: int) -> np.ndarray:
                if img is None:
                    return np.zeros((height, width, 3), dtype=np.uint8)
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                if img.shape[0] != height or img.shape[1] != width:
                    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                return img

            left = _force_bgr(left, left_w, pane_h)
            mid = _force_bgr(mid, mid_w, pane_h)
            right = _force_bgr(right, right_w, pane_h)

            canvas = np.zeros((pane_h, left_w + mid_w + right_w, 3), dtype=np.uint8)
            canvas[:, 0:left_w] = left
            canvas[:, left_w : left_w + mid_w] = mid
            canvas[:, left_w + mid_w : left_w + mid_w + right_w] = right
            cv2.imshow(win, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('a'):
                import time

                ts = time.strftime('%Y%m%d_%H%M%S')
                out_path = os.path.join(CACHE_DIR, 'captured', f'cam_{ts}_{int(time.time()*1000)%1000:03d}.jpg')
                cv2.imwrite(out_path, frame)
                if engine.add_frame_to_database(frame, out_path):
                    added_paths.append(out_path)
                else:
                    print('No features found in frame; not added')

    finally:
        cap.release()
        cv2.destroyAllWindows()


def pick_query_image(initial_dir: str) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        print("tkinter not available; use --mpl-gui or --camera")
        return None

    root = tk.Tk()
    root.withdraw()
    try:
        path = filedialog.askopenfilename(
            title="Select query image",
            initialdir=initial_dir,
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*"),
            ],
        )
        return path or None
    finally:
        root.destroy()


def pick_query_image_matplotlib(image_paths: list[str], max_images: int = 30, cols: int = 6) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; cannot open picker")
        return None

    shown_paths = image_paths[:max_images]
    if not shown_paths:
        return None

    n = len(shown_paths)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).reshape(-1)

    selected = {"path": None}

    for ax, path in zip(axes, shown_paths):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            ax.axis("off")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(os.path.basename(path), fontsize=8)
        ax.axis("off")
        ax._img_path = path  # store path on the axes

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Click an image to select query (close window to cancel)", fontsize=12)

    def _on_click(event):
        ax = event.inaxes
        if ax is None:
            return
        path = getattr(ax, "_img_path", None)
        if path:
            selected["path"] = path
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", _on_click)
    plt.tight_layout()
    plt.show()

    return selected["path"]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", default="resource/sfm_images")
    parser.add_argument("--query", default=None, help="Path to query image")
    parser.add_argument("--gui", action="store_true", help="Open file picker for query image")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--no-viz", action="store_true", help="Disable matplotlib visualization")
    parser.add_argument("--mpl-gui", action="store_true", help="Pick query by clicking an image grid (matplotlib)")
    parser.add_argument("--camera", action="store_true", help="Live camera mode (OpenCV GUI)")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index (default: 0)")

    args = parser.parse_args()

    K = 8
    LEVELS = 2

    engine = ImageSearchEngine(k=K, levels=LEVELS)
    engine.build_database(args.image_folder, force_rebuild=args.force_rebuild)

    if args.camera:
        run_camera_stream(engine, top_k=args.top_k, camera_index=args.camera_index)
        return


    image_paths = _list_images(args.image_folder)
    if not image_paths:
        print(f"No images found in {args.image_folder}")
        return

    if args.mpl_gui:
        picked = pick_query_image_matplotlib(image_paths, max_images=60, cols=6)
        if picked is None:
            print("No image selected; exiting.")
            return
        query = picked
    elif args.gui:
        picked = pick_query_image(args.image_folder)  # tkinter version (optional)
        if picked is None:
            print("No image selected; exiting.")
            return
        query = picked
    else:
        query = args.query if args.query is not None else image_paths[0]

    print(f"\nQuery: {os.path.basename(query)}")

    results = engine.search(query, top_k=args.top_k)
    print("Top results:")
    for rank, (path, score) in enumerate(results, start=1):
        print(f"{rank}. {os.path.basename(path)}  score={score:.4f}")

    if (not args.no_viz) and results:
        engine.visualize_results(query, results, max_display=args.top_k)


if __name__ == "__main__":
    main()
