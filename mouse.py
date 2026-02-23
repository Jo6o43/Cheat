import ctypes
import numpy as np
from windMouse import wind_mouse

# Windows Constants
MOUSEEVENTF_MOVE = 0x0001 

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_void_p),
    ]

class INPUT_I(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]

class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", INPUT_I)]

class MouseController:
    def __init__(self):
        self.screen_w = ctypes.windll.user32.GetSystemMetrics(0)
        self.screen_h = ctypes.windll.user32.GetSystemMetrics(1)
        # In FPS, we start from the crosshair (center)
        self.center_x = self.screen_w // 2
        self.center_y = self.screen_h // 2
        self.keys = {"LBUTTON": 0x01, "RBUTTON": 0x02, "ALT": 0x12}

    def _relative_move_callback(self, new_x, new_y):
        """
        This is the move_mouse lambda.
        It calculates how much the mouse moved since the last WindMouse step.
        """
        # Calculate the delta from the last position
        dx = new_x - self.last_x
        dy = new_y - self.last_y
        
        # Send the delta to Windows
        ii_ = INPUT_I()
        ii_.mi = MOUSEINPUT(dx, dy, 0, MOUSEEVENTF_MOVE, 0, ctypes.c_void_p(0))
        input_obj = INPUT(0, ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(input_obj), ctypes.sizeof(input_obj))
        
        # Update last position for the next step
        self.last_x, self.last_y = new_x, new_y

    def move_to(self, dest_x, dest_y, activation_key="ALT"):
        """Calls your WindMouse function with a delta-tracking callback."""
        
        # 1. Hotkey Check
        vk = self.keys.get(activation_key.upper(), activation_key)
        if not (ctypes.windll.user32.GetAsyncKeyState(int(vk)) & 0x8000):
            return

        # 2. Setup state for the callback
        self.last_x = self.center_x
        self.last_y = self.center_y

        # 3. Call your WindMouse function
        # We pass our _relative_move_callback to turn absolute steps into relative deltas
        wind_mouse(
            start_x=self.center_x, 
            start_y=self.center_y, 
            dest_x=dest_x, 
            dest_y=dest_y, 
            move_mouse=self._relative_move_callback
        )