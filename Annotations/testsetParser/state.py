import reflex as rx

class State(rx.State):
    """The app state."""
    video_url: str = ""
    current_frame: int = 0
    
    def set_video_url(self, url: str):
        """Set the video URL."""
        self.video_url = url
    
    def update_frame(self, frame: int):
        """Update the current frame number."""
        self.current_frame = frame
