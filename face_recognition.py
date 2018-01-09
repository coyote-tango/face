# Import OpenCV2 for image processing
import cv2
import time

# Import numpy for matrices calculations
import numpy as np

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()
found = False
# Load the trained mode
recognizer.read('trainer/trainer.yml')

# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture
cam = cv2.VideoCapture(0)

# Loop
while True:
	# Read the video frame
	ret, im =cam.read()

	# Convert the captured frame into grayscale
	gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

	# Get all face from the video frame
	faces = faceCascade.detectMultiScale(gray, 1.2,5)

	# For each face in faces
	for(x,y,w,h) in faces:

		# Create rectangle around the face
		cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

		# Recognize the face belongs to which ID
		Id, conf = recognizer.predict(gray[y:y+h,x:x+w])

		# Check the ID if exist 
		if(Id == 1 and conf < 60):
			print(Id, conf)
			Id = "Juan Pablo"
			found = True
		#If not exist, then it is Unknown
		elif(Id == 2 and conf < 60):
			print(Id, conf)
			Id = "Omar"
			found = True
		else:
			print(Id, conf)
			Id = "Unknown"

		# Put text describe who is in the picture
		cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
		cv2.putText(im, str(Id), (x,y-40), font, 2, (255,255,255), 3)

	# Display the video frame with the bounded rectangle
	cv2.imshow('im',im)

	# If 'q' is pressed, close program
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break

	if (found):
		DELAY = 0.02
		USE_CAM = 1
		IS_FOUND = 0

		MORPH = 7
		CANNY = 250
		##################
		# 420x600 oranı 105mmx150mm gerçek boyuttaki kağıt için
		_width  = 600.0
		_height = 420.0
		_margin = 0.0
		##################

		if USE_CAM: video_capture = cv2.VideoCapture(0)

		corners = np.array(
			[
				[[  		_margin, _margin 			]],
				[[ 			_margin, _height + _margin  ]],
				[[ _width + _margin, _height + _margin  ]],
				[[ _width + _margin, _margin 			]],
			]
		)

		pts_dst = np.array( corners, np.float32 )

		while True :

			if USE_CAM :
				ret, rgb = video_capture.read()
			else :
				ret = 1
				rgb = cv2.imread( "opencv.jpg", 1 )

			if ( ret ):

				gray = cv2.cvtColor( rgb, cv2.COLOR_BGR2GRAY )

				gray = cv2.bilateralFilter( gray, 1, 10, 120 )

				edges  = cv2.Canny( gray, 10, CANNY )

				kernel = cv2.getStructuringElement( cv2.MORPH_RECT, ( MORPH, MORPH ) )

				closed = cv2.morphologyEx( edges, cv2.MORPH_CLOSE, kernel )

				_,contours, h = cv2.findContours( closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

				for cont in contours:

					# Küçük alanları pass geç
					if cv2.contourArea( cont ) > 5000 :

						arc_len = cv2.arcLength( cont, True )

						approx = cv2.approxPolyDP( cont, 0.1 * arc_len, True )

						if ( len( approx ) == 4 ):
							IS_FOUND = 1
							#M = cv2.moments( cont )
							#cX = int(M["m10"] / M["m00"])
							#cY = int(M["m01"] / M["m00"])
							#cv2.putText(rgb, "Center", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

							pts_src = np.array( approx, np.float32 )

							h, status = cv2.findHomography( pts_src, pts_dst )
							out = cv2.warpPerspective( rgb, h, ( int( _width + _margin * 2 ), int( _height + _margin * 2 ) ) )

							cv2.drawContours( rgb, [approx], -1, ( 255, 0, 0 ), 2 )

						else : pass

				#cv2.imshow( 'closed', closed )
				#cv2.imshow( 'gray', gray )
				

				cv2.namedWindow( 'rgb')
				cv2.imshow( 'rgb', rgb )

				if IS_FOUND :
					cv2.namedWindow( 'out')
					cv2.imshow( 'out', out )

				if cv2.waitKey(27) & 0xFF == ord('q') :
					break

				if cv2.waitKey(99) & 0xFF == ord('c') :
					current = str( time.time() )
					cv2.imwrite( 'ocvi_' + current + '_edges.jpg', edges )
					cv2.imwrite( 'ocvi_' + current + '_gray.jpg', gray )
					cv2.imwrite( 'ocvi_' + current + '_org.jpg', rgb )
					print('Pictures saved')

				time.sleep( DELAY )

			else :
				print ('Stopped')
				break

		if USE_CAM : video_capture.release()
		cv2.destroyAllWindows()
	# end

# Stop the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()
