# Taking pictures
from picamera import PiCamera
from time import sleep

camera = PiCamera()
camera.resolution = (1024,768)


camera.start_preview()
sleep(5)
camera.capture('image.jpg')
camera.stop_preview()

# Taking videos
camera = PiCamera()
camera.resolution = (640,480)

camera.start_preview()
camera.start_recording('video.h264')
camera.wait_recording(120)
camera.stop_recording()
camera.stop_preview()