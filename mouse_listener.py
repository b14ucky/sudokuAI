from pynput import mouse


class MouseListener:
    def __init__(self):
        self.mouse_clicked = False
        self.listener = mouse.Listener(on_click=self.on_click)

    def on_click(self, x, y, button, pressed):
        if pressed and button == mouse.Button.left:
            self.mouse_clicked = True
            self.listener.stop()

    def wait_for_click(self):
        print("Waiting for the user to click the left mouse button...")
        self.listener.start()
        self.listener.join()
