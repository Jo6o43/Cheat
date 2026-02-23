import cv2
import numpy as np
import capture
import display
import mouse


def create_trackbar_window(name: str, detector: capture.ColorDetector):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    # Initial values from detector
    lh, ls, lv = int(detector.lower_color[0]), int(detector.lower_color[1]), int(detector.lower_color[2])
    hh, hs, hv = int(detector.upper_color[0]), int(detector.upper_color[1]), int(detector.upper_color[2])
    cv2.createTrackbar('H Low', name, lh, 179, lambda x: None)
    cv2.createTrackbar('S Low', name, ls, 255, lambda x: None)
    cv2.createTrackbar('V Low', name, lv, 255, lambda x: None)
    cv2.createTrackbar('H High', name, hh, 179, lambda x: None)
    cv2.createTrackbar('S High', name, hs, 255, lambda x: None)
    cv2.createTrackbar('V High', name, hv, 255, lambda x: None)
    cv2.createTrackbar('Search Size', name, detector.search_size, 1000, lambda x: None)


def read_trackbar_values(name: str):
    lh = cv2.getTrackbarPos('H Low', name)
    ls = cv2.getTrackbarPos('S Low', name)
    lv = cv2.getTrackbarPos('V Low', name)
    hh = cv2.getTrackbarPos('H High', name)
    hs = cv2.getTrackbarPos('S High', name)
    hv = cv2.getTrackbarPos('V High', name)
    sz = cv2.getTrackbarPos('Search Size', name)
    return (np.array([lh, ls, lv], dtype=np.int32), np.array([hh, hs, hv], dtype=np.int32), max(20, sz))


def main():
    brain = capture.ColorDetector()
    painter = display.Display(scale=1.5)
    aimbot = mouse.MouseController()

    tuning_win = "Color Tuning"
    create_trackbar_window(tuning_win, brain)
    cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)

    print("Running... Press 'q' in the window to stop. Use the 'Color Tuning' window to adjust HSV and search size.")

    while True:
        # Update thresholds from trackbars
        lower, upper, ssz = read_trackbar_values(tuning_win)
        brain.set_thresholds(lower, upper)
        brain.set_search_size(ssz)

        frame, people, mask = brain.get_data()

        if people and people[0]["head"]:
            hx, hy = people[0]["head"]
            aimbot.move_to(hx, hy, activation_key="ALT")

        painter.show(frame, people)

        # Show the binary mask in a separate window with the same pixel size
        try:
            scaled_frame = painter.get_scaled_frame(frame)
            h, w = scaled_frame.shape[:2]
            scaled_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('Mask', scaled_mask)
            try:
                cv2.resizeWindow('Mask', w, h)
            except Exception:
                pass
        except Exception:
            pass

        if painter.should_quit():
            break

    painter.close()


if __name__ == "__main__":
    main()