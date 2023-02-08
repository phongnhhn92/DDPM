import os
import cv2

def sortFiles(files):
    return sorted(files, key=lambda x: int(x.split('.')[0]))

def createVideoFromImageFolder(image_folder, video_name):   

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = sortFiles(images)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(video_name, fourcc, 1., (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    image_folder = './samples'
    video_name = 'video.mp4'
    createVideoFromImageFolder(image_folder,video_name)