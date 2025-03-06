import tkinter as tk
from tkinter import filedialog, messagebox
import os
import cv2
import numpy as np

class ImageBrowser:
    def __init__(self, root):
        self.root = root
        self.root.title("JPG Image Clicker")

        self.folder_path = ""
        self.image_paths = []
        self.current_image_index = 0

        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()

        self.prev_button = tk.Button(root, text="Previous", command=self.show_previous_image)
        self.prev_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(root, text="Next", command=self.show_next_image)
        self.next_button.pack(side=tk.RIGHT)

        self.open_button = tk.Button(root, text="Open Folder", command=self.open_folder)
        self.open_button.pack()

        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.clicks = []
        self.image_label = tk.Label(root, text="", font=("Arial", 12))
        self.image_label.pack()

    def open_folder(self):
        self.folder_path = filedialog.askdirectory()
        if self.folder_path:
            self.image_paths = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.lower().endswith(".jpg")]
            self.current_image_index = 0
            self.load_image()

    def load_image(self):
        if self.image_paths:
            image_path = self.image_paths[self.current_image_index]
            image = cv2.imread(image_path)
            self.show_image(image, image_path)

    def show_image(self, img, path):
        # Resize the image to 1024x1024 while maintaining aspect ratio
        original_height, original_width = img.shape[:2]
        target_height = 1024
        target_width = 1024

        aspect_ratio = original_width / original_height
        if original_height > original_width:
            new_width = int(target_height * aspect_ratio)
            new_height = target_height
        else:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)

        resized_img = cv2.resize(img, (new_width, new_height))

        # Center the image in the 1024x1024 window
        canvas_img = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
        start_x = (target_width - new_width) // 2
        start_y = (target_height - new_height) // 2
        canvas_img[start_y:start_y + new_height, start_x:start_x + new_width] = resized_img

        self.photo = tk.PhotoImage(master=self.root, data=cv2.imencode('.png', canvas_img)[1].tobytes())
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

        cv2.imshow('Image', canvas_img)
        cv2.setMouseCallback('Image', self.mouse_callback)

        # Update the image label with the file name
        self.image_label.config(text=os.path.basename(path))

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x, y = map(int, [x, y])
            img_name = os.path.basename(self.image_paths[self.current_image_index])
            self.clicks.append((img_name, x, y))
            with open("clicks.txt", "a") as file:
                file.write(f"{img_name}, {x}, {y}\n")
            print(f"Clicked at ({x}, {y}) on {img_name}")

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image()

    def show_next_image(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.load_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageBrowser(root)
    root.mainloop()
