import cv2
from YOLO_Prediction import YOLO_PRED
import matplotlib.pyplot as plt


# def plot_image(image, save_path=None):
#     plt.figure(figsize=(12, 8))  # Ensure the figure is large
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
#     plt.axis('off')  # Hide the axis
#     if save_path:
#         plt.savefig(f'{save_path}/prediction.png', bbox_inches='tight')  # Save without margins
#     plt.show()

# yolo = YOLO_PRED('./Model/weights/best.onnx', 'data.yaml')
# img = cv2.imread('./Us.jpeg')
# save_path = './Predictions/'

# img_pred = yolo.predictions(img)
# plot_image(img_pred, save_path)

# cv2.imshow('pred_image', img_pred)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

yolo = YOLO_PRED('./Model/weights/best.onnx', 'data.yaml')

def plot_frame(frame):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.01)

cap = cv2.VideoCapture('video.mp4')
while True:
    ret, frame = cap.read()
    if ret == False:
        print('unable to read video')
        break

    pred_image = yolo.predictions(frame)
    # # plot_frame(pred_image)
    # if cv2.waitKey(1) == 27:  # ESC key to exit
    #     break
    cv2.imshow('YOLO', pred_image)
    if cv2.waitKey(1) == 27:
        break

cap.release()



# plt.close()
# cv2.destroyAllWindows()

