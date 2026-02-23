import capture
import display
import mouse

def main():
    brain = capture.Capture()
    painter = display.Display()
    aimbot = mouse.MouseController()

    print("Running... Press 'q' in the window to stop.")

    while True:
        frame, people = brain.get_data()

        if people and people[0]["head"]:
            hx, hy = people[0]["head"]
            aimbot.move_to(hx, hy, activation_key="ALT")
            print(f"Moved mouse to: ({hx}, {hy})")

        painter.show(frame, people)

        if painter.should_quit():
            break

    painter.close()

if __name__ == "__main__":
    main()