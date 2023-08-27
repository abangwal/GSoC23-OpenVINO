from flask import Flask,render_template,Response,request
import cv2
import numpy as np
import time 
import pathlib
import openvino.runtime as ov
import pickle
import datetime
from openvino.runtime import Core
from utils import *

global prev_data,readings

plotted_img = [np.zeros([1080,1920,3],dtype=np.int8) for i in range(5)]

prev_data = {0:{},1:{},2:{},3:{}}

with open('configs.pkl','rb') as file:
	d = pickle.load(file)

readings = {}

for i in range(len(list(d.keys()))):
	readings[i] = {'time':[0],'reading':[0]}

for i in range(4):
	try:
		prev_data[i]['interval'] = d[i+1]['interval']
	except:
		prev_data[i]['interval'] = 60
	prev_data[i]['prev_plot'] = plotted_img[i]


def segmentor_callback(infer_req, info):
	global prev_data, readings
	print(readings)
	frame,results,config,idx = info
	pred_seg = infer_req.get_output_tensor(0).data
	processed_seg = []
	for i in pred_seg:
		processed_seg.append(cv2.resize(i,(512,512),cv2.INTER_AREA))
	pred = np.argmax(processed_seg,axis=3)

	# Getting Reading from predicted Maps
	pred = erode(pred,3)
	rectangle_meters = circle_to_rectangle(pred)
	line_scales, line_pointers = rectangle_to_line(rectangle_meters)
	binaried_scales = mean_binarization(line_scales)
	binaried_pointers = mean_binarization(line_pointers)
	scale_locations = locate_scale(binaried_scales)
	pointer_locations = locate_pointer(binaried_pointers)
	pointed_scales = get_relative_location(scale_locations, pointer_locations)
	meter_readings = calculate_reading(pointed_scales,meter_config=config)

	# Plotting reading and BBOXs on image
	readings[idx]['reading'] += meter_readings
	readings[idx]['time'] += [time.asctime(time.localtime()).split()[3]]
	if len(readings[idx]['reading']) >=4:
		readings[idx]['reading'].pop(0)
		readings[idx]['time'].pop(0)

	prev_data[idx]['prev_plot'] = plot_result(frame,meter_readings,results)

def detector_callback(infer_req , info):
	global prev_data
	frame,config,idx = info
	result_ratio = infer_req.get_output_tensor(1).data[0]
	scores = infer_req.get_output_tensor(4).data[0]
	selected_rr = []

	for i in range(len(scores)):
		if scores[i]>0.5:
			selected_rr.append(result_ratio[i])
		else:
			break
	try:
		results = np.multiply(selected_rr,[1082,1920,1082,1920]).astype(np.int64)
	except:
		plotted_img[idx] = frame
		return None
	# Cropping Meters
	roi_imgs,loc = roi_crop(frame,results,1,1)
	# Preprocess uneven cropped imgs to 256,256
	crop_img = []
	for roi_img in roi_imgs:
		resized = cv2.resize(roi_img,(256,256),cv2.INTER_AREA)
		crop_img.append(resized)

	if len(crop_img) > 0:
		segmentor_input = np.array(crop_img)
		async_segmentor.start_async({s_input_layer_ir.any_name : segmentor_input},(frame,results,config,idx))
	else:
		prev_data[idx]['prev_plot'] = frame


core = Core()
# read converted model

detector_ir = core.read_model(model='IR/detector_IR/detector.xml')
segmentor_ir = core.read_model(model='IR/segmentor_IR/segmentor.xml')

detector = core.compile_model(detector_ir,'CPU',{"PERFORMANCE_HINT":"LATENCY","NUM_STREAMS":3, 'INFERENCE_NUM_THREADS':6, 'AFFINITY':'NUMA'})
segmentor = core.compile_model(segmentor_ir,'CPU',{"PERFORMANCE_HINT":"LATENCY","NUM_STREAMS":3, 'INFERENCE_NUM_THREADS':6, 'AFFINITY':'NUMA'})

# creating Async inference models
async_detector = ov.AsyncInferQueue(detector,4)
async_segmentor = ov.AsyncInferQueue(segmentor,4)

async_detector.set_callback(detector_callback)
async_segmentor.set_callback(segmentor_callback)

d_input_layer_ir = detector_ir.input(0)
s_input_layer_ir = segmentor_ir.input(0)

sources = ['static/vid1.mp4','static/vid2.mp4','static/vid3.mp4','static/vid4.mp4']
configs = [{'scale_interval_value':60/60},{'scale_interval_value':10/20},
			{'scale_interval_value':60/60},{'scale_interval_value':10/20}]

def gen_frames(idx):
	idx = int(idx)
	config = configs[idx]
	global prev_data,readings
	cam = cv2.VideoCapture(sources[idx])
	current_time = 0
	while True:
		time.sleep(1)
		current_time += 1
		for i in prev_data:
			if ((current_time % prev_data[i]['interval']) == 0 and idx == i) or (current_time <= 2) :
				_,f = cam.read()
				if _:
					frame = cv2.resize(f,(1920,1080),cv2.INTER_AREA)
					frame = frame.reshape(1,1080,1920,3)
					async_detector.start_async({d_input_layer_ir.any_name : frame},(frame[0],config,idx))
					ret, buffer = cv2.imencode('.jpg', prev_data[idx]['prev_plot'])
					frame = buffer.tobytes()
					yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
				else:
					break

	ret, buffer = cv2.imencode('.jpg', prev_data[idx]['prev_plot'])
	frame = buffer.tobytes()
	yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
	async_detector.wait_all()
	async_segmentor.wait_all()


app = Flask(__name__)


@app.route('/')
def index():
	global readings
	return render_template("index.html",prev_data=prev_data,readings=readings)

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    return Response(gen_frames(idx=camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/reading_feed/<camera_id>')
def reading_feed(camera_id):
	global readings
	return readings[int(camera_id)]

@app.route('/settings',methods=['POST','GET'])
def settings():
	if request.method == 'POST':
		try:
			with open('configs.pkl','rb') as file:
				prev_configs = pickle.load(file)

			while len(prev_configs) < 5:
				prev_configs[len(prev_configs)+1] = [0,0]

			n = request.form['number_of_cameras']
			return render_template('settings.html',n=int(n),prev_configs=prev_configs)
		
		except:

			configs = {}
			data = request.form
			print(data)
			for i in range(1,(len(data)//3)+1):
				rtd = data[f'refresh_time{i}'].split(':')
				try:
					rt = int(rtd[0])*60 + int(rtd[1])*1
				except:
					rt = 60
				configs[i]={
							'meter_range' : int(data[f'meter_range{i}']),
							'meter_interval' : int(data[f'n_interval{i}']),
							'interval' : rt
							}

			with open('configs.pkl','wb') as file:
				pickle.dump(configs,file)

			return render_template('settings.html',n=int(0),data=configs,prev_configs='')

	else:
		return render_template('settings.html',n=0,prev_configs='')

if __name__ == '__main__':
	app.run(debug=True)
