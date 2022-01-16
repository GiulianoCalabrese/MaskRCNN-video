import torch
import torchvision
import cv2
import time
import argparse
from PIL import Image
from utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, 
                    help='path to the input data')
parser.add_argument('-t', '--threshold', default=0.965, type=float,
                    help='score threshold for discarding detection')
args = vars(parser.parse_args())

# initialize the model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, 
                                                           num_classes=91)
# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
# load the model on to the computation device and set to eval mode
model.to(device).eval()

# transform to convert the image to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# initialize the video stream and pointer to output video file
print(args["input"])
vs = cv2.VideoCapture(args["input"])
#vs = cv2.VideoCapture('/home/giuliano/samples/videos/test.mp4')
writer = None
# try to determine the total number of frames in the video file
try:
	total= int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
	print("[INFO] {} total frames in video".format(total))
# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	total = -1

writer = None
nb_frame = 0
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	nb_frame += 1
	print(nb_frame)
	if not grabbed:
		break		
		
	start = time.time()
	# keep a copy of the original image for OpenCV functions and applying masks
	orig_image = frame.copy()
	# transform the image
	image = transform(frame)
	# add a batch dimension
	image = image.unsqueeze(0).to(device)
	masks, boxes, labels = get_outputs(image, model, args['threshold'])
	result = draw_segmentation_map(orig_image, masks, boxes, labels)
	end = time.time()			
	# check if the video writer is None
	if writer is None:
		print(grabbed)
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter("output.avi", fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)
		# some information on processing single frame
		print(total)
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))
	# write the output frame to disk
	writer.write(result)
	# Display the resulting frame   
	#cv2.imshow('frame',result)
        # Press Q on keyboard to stop recording
	#if cv2.waitKey(1) & 0xFF == ord('q'):
		#break
		
# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()

