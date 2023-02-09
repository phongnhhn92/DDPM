import os
import cv2
import moviepy.editor as mpy

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

def createVisfromImageFolder(image_folder, filename, saveGIF=True, saveVideo=True):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = sortFiles(images)
    images = [os.path.join(image_folder, img) for img in images]
    clip = mpy.ImageSequenceClip(images, fps=24)
    if saveGIF:
        clip.write_gif(os.path.join(image_folder,f'{filename}.gif'), fps=5)
    if saveVideo:
        clip.write_videofile(os.path.join(image_folder,f'{filename}.mp4'), fps=5)

if __name__ == "__main__":
    image_folder = './samples/exp_name/0/generated'
    video_name = 'video.mp4'
    createVisfromImageFolder(image_folder, 'video', saveGIF=True, saveVideo=True)