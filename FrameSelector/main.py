import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageGrab
import tempfile
import boto3
from dotenv import load_dotenv
import os
import io

class VideoPlayer:
    def __init__(self, root):
        load_dotenv()
        self.root = root
        self.root.title("S3 Video Frame Selector")
        
        # Main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # URL input frame
        self.url_frame = ttk.Frame(self.main_frame)
        self.url_frame.grid(row=0, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E))
        
        # URL label and entry
        ttk.Label(self.url_frame, text="S3 URL:").pack(side=tk.LEFT, padx=5)
        self.url_entry = ttk.Entry(self.url_frame, width=60)
        self.url_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Load button
        self.load_button = ttk.Button(self.url_frame, text="Load Video", command=self.load_video)
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        # Video display
        self.canvas = tk.Canvas(self.main_frame, width=800, height=600)
        self.canvas.grid(row=1, column=0, columnspan=3)
        
        # Frame counter and controls frame
        self.controls_frame = ttk.Frame(self.main_frame)
        self.controls_frame.grid(row=2, column=0, columnspan=3, pady=5)
        
        # Frame counter
        self.frame_label = ttk.Label(self.controls_frame, text="Frame: 0")
        self.frame_label.pack(side=tk.LEFT, padx=5)
        
        # Copy frame button
        self.copy_button = ttk.Button(self.controls_frame, text="Copy Frame", command=self.copy_frame)
        self.copy_button.pack(side=tk.LEFT, padx=5)
        
        # Slider
        self.slider = ttk.Scale(self.main_frame, orient="horizontal", length=800)
        self.slider.grid(row=3, column=0, columnspan=3, pady=5)
        self.slider.bind("<ButtonRelease-1>", self.slider_changed)
        
        # Initialize video
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.temp_file_path = None
        self.current_pil_image = None  # Store the current PIL image
        
        # Bind keys
        self.root.bind('<Left>', lambda e: self.prev_frame())
        self.root.bind('<Right>', lambda e: self.next_frame())
        
        # Bind cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.cleanup)

    def cleanup(self):
        """Clean up resources before closing"""
        if self.cap is not None:
            self.cap.release()
        
        # Remove temporary file if it exists
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            try:
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                os.unlink(self.temp_file_path)
            except Exception as e:
                print(f"Error cleaning up temporary file: {e}")
        
        self.root.destroy()

    def cleanup_current_video(self):
        """Clean up current video resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            try:
                os.unlink(self.temp_file_path)
            except Exception as e:
                print(f"Error cleaning up temporary file: {e}")
            self.temp_file_path = None

    def load_video(self):
        # Cleanup previous video if exists
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            try:
                os.unlink(self.temp_file_path)
            except Exception as e:
                print(f"Error removing previous temporary file: {e}")
            self.temp_file_path = None

        # Get S3 credentials from .env file
        bucket_name = os.getenv('S3_BUCKET')
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

        # Get video URL from entry
        video_url = self.url_entry.get().strip()
        if not video_url:
            tk.messagebox.showerror("Error", "Please enter a video URL")
            return

        # Extract key from URL (assuming URL format: https://bucket-name.s3.region.amazonaws.com/key)
        try:
            video_key = video_url.split('.amazonaws.com/')[-1]
        except:
            tk.messagebox.showerror("Error", "Invalid S3 URL format")
            return

        # Validate credentials
        if not all([bucket_name, aws_access_key, aws_secret_key]):
            tk.messagebox.showerror("Error", "Missing S3 credentials. Please check your .env file.")
            return

        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key
            )
            
            # Create temporary file with unique name
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            self.temp_file_path = temp_file.name
            temp_file.close()  # Close the file handle immediately
            
            try:
                s3_client.download_file(bucket_name, video_key, self.temp_file_path)
                self.cap = cv2.VideoCapture(self.temp_file_path)
                
                if not self.cap.isOpened():
                    tk.messagebox.showerror("Error", "Failed to open video file")
                    self.cleanup_current_video()
                    return
                    
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.slider.configure(from_=0, to=self.total_frames-1)
                self.current_frame = 0
                self.show_frame()
            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to download video: {str(e)}")
                self.cleanup_current_video()
                
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to connect to S3: {str(e)}")
            self.cleanup_current_video()

    def show_frame(self):
        if self.cap is None:
            return
            
        if not self.cap.isOpened():
            return
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        
        if ret:
            # Convert frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame to fit canvas while maintaining aspect ratio
            height, width = frame.shape[:2]
            canvas_width = 800
            canvas_height = 600
            scale = min(canvas_width/width, canvas_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert to PIL Image and store it
            self.current_pil_image = Image.fromarray(frame)
            
            # Convert to PhotoImage for display
            self.photo = ImageTk.PhotoImage(image=self.current_pil_image)
            
            # Update canvas and label
            self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo, anchor=tk.CENTER)
            self.frame_label.configure(text=f"Frame: {self.current_frame}")
            self.slider.set(self.current_frame)

    def next_frame(self):
        if self.cap is not None and self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.show_frame()

    def prev_frame(self):
        if self.cap is not None and self.current_frame > 0:
            self.current_frame -= 1
            self.show_frame()

    def slider_changed(self, event):
        if self.cap is not None:
            self.current_frame = int(self.slider.get())
            self.show_frame()

    def copy_frame(self):
        """Copy the current frame to clipboard"""
        if self.current_pil_image is None:
            tk.messagebox.showwarning("Warning", "No frame available to copy")
            return
            
        try:
            # Create a temporary buffer
            output = io.BytesIO()
            self.current_pil_image.save(output, 'BMP')
            data = output.getvalue()[14:]  # Remove BMP header
            output.close()
            
            # Clear clipboard and append new image
            self.root.clipboard_clear()
            self.root.clipboard_append(data)
            
            # Show success message
            self.root.status_message = tk.StringVar()
            status_label = ttk.Label(self.controls_frame, textvariable=self.root.status_message)
            status_label.pack(side=tk.LEFT, padx=5)
            self.root.status_message.set("Frame copied!")
            self.root.after(2000, lambda: self.root.status_message.set(""))  # Clear message after 2 seconds
            
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to copy frame: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoPlayer(root)
    root.mainloop()
