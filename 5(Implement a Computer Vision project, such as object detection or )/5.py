from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt


def main():
    # Load YOLO models
    det_model = YOLO("yolov8n.pt")       # Object Detection model
    seg_model = YOLO("yolov8n-seg.pt")   # Segmentation model

    # Load image
    img = cv2.imread(r"C:\Users\Rohsn Chimbaikar\Downloads\IA-main\IA-main\Practical 5\new.jpeg") #or downlaod any image and change the name here

    if img is None:
        print("ERROR: streets.jpg not found in the folder.")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Object Detection
    det_res = det_model(img_rgb)[0].plot()

    # Image Segmentation
    seg_res = seg_model(img_rgb)[0].plot()

    # Display Detection Result
    plt.figure(figsize=(10, 6))
    plt.title("Object Detection")
    plt.imshow(det_res)
    plt.axis("off")
    plt.show()

    # Display Segmentation Result
    plt.figure(figsize=(10, 6))
    plt.title("Image Segmentation")
    plt.imshow(seg_res)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
