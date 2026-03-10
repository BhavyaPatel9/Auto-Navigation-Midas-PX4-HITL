import time
import numpy as np
import airsim
import cv2
import tensorrt as trt
import ctypes
from PIL import Image
from math import radians, cos, sin, degrees, sqrt
from pymavlink import mavutil
import matplotlib.pyplot as plt


# ================= PARAMETERS =================

SAFE_THRESHOLD = 0.55
NUM_SECTORS = 91
MIN_VALLEY_WIDTH = 18

BASE_SPEED = 1.5
MIN_SPEED = 0.5
CLIMB_SPEED = -2.0

MAX_DISTANCE = 45

INPUT_SIZE = 224
DT = 0.1

WINDOWS_IP = "192.168.68.62"
ENGINE_PATH = "/home/swayaan/midas-small_fp16.engine"

DISPLAY_W = 480
DISPLAY_H = 360


# ================= CUDA =================

cudart = ctypes.CDLL("libcudart.so")

cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2


def cuda_malloc(size):
    ptr = ctypes.c_void_p()
    cudart.cudaMalloc(ctypes.byref(ptr), size)
    return ptr.value


def cuda_htod(dst, src, size):
    cudart.cudaMemcpy(ctypes.c_void_p(dst),
                      src.ctypes.data_as(ctypes.c_void_p),
                      size,
                      cudaMemcpyHostToDevice)


def cuda_dtoh(dst, src, size):
    cudart.cudaMemcpy(dst.ctypes.data_as(ctypes.c_void_p),
                      ctypes.c_void_p(src),
                      size,
                      cudaMemcpyDeviceToHost)


# ================= TensorRT =================

class TRT:

    def __init__(self, engine_path):

        logger = trt.Logger(trt.Logger.ERROR)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self.inputs=[]
        self.outputs=[]
        self.bindings=[]

        for i in range(self.engine.num_bindings):

            shape=self.engine.get_binding_shape(i)
            dtype=trt.nptype(self.engine.get_binding_dtype(i))

            host=np.empty(shape,dtype=dtype)
            device=cuda_malloc(host.nbytes)

            self.bindings.append(device)

            buf={"host":host,"device":device,"bytes":host.nbytes}

            if self.engine.binding_is_input(i):
                self.inputs.append(buf)
            else:
                self.outputs.append(buf)

    def infer(self,x):

        np.copyto(self.inputs[0]["host"],
                  x.astype(self.inputs[0]["host"].dtype))

        cuda_htod(self.inputs[0]["device"],
                  self.inputs[0]["host"],
                  self.inputs[0]["bytes"])

        self.context.execute_v2(self.bindings)

        cuda_dtoh(self.outputs[0]["host"],
                  self.outputs[0]["device"],
                  self.outputs[0]["bytes"])

        return self.outputs[0]["host"]


# ================= PREPROCESS =================

def preprocess(img):

    img = img.resize((INPUT_SIZE, INPUT_SIZE))
    img = np.array(img).astype(np.float32) / 255.0

    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])

    img = (img - mean) / std
    img = img.transpose(2,0,1)
    img = np.expand_dims(img,0)

    return img


# ================= DEPTH =================

def get_depth_map():

    resp = client.simGetImages([
        airsim.ImageRequest("front_center",
                            airsim.ImageType.Scene,
                            False,False)
    ])[0]

    img = np.frombuffer(resp.image_data_uint8,dtype=np.uint8)

    if img.size == 0:
        return np.ones((224,224)), np.zeros((224,224,3),dtype=np.uint8)

    img = img.reshape(resp.height,resp.width,3)

    rgb = img.copy()

    pil_img = Image.fromarray(rgb)

    inp = preprocess(pil_img)

    # -------- RAW MIDAS DEPTH --------
    raw_depth = model.infer(inp).squeeze()

    raw_norm = (raw_depth - raw_depth.min())/(raw_depth.max()-raw_depth.min()+1e-6)

    # -------- PROCESSED DEPTH FOR NAVIGATION --------
    depth = 1.0 - raw_norm

    depth = cv2.GaussianBlur(depth,(9,9),0)

    depth = cv2.dilate(depth,np.ones((11,11),np.uint8))

    return depth, raw_norm, rgb


# ================= HISTOGRAM =================

def compute_histogram(depth):

    h,w = depth.shape

    sector_w = w//NUM_SECTORS

    region = depth[int(0.30*h):int(0.80*h),:]

    hist = np.zeros(NUM_SECTORS)

    for i in range(NUM_SECTORS):

        col = region[:,i*sector_w:(i+1)*sector_w]

        v = col[col>0]

        hist[i] = np.percentile(v,30) if v.size>0 else 1.0

    angles = np.linspace(-radians(90),radians(90),NUM_SECTORS)

    return hist,angles


# ================= VALLEY =================

def find_best_sector(hist,angles):

    valid = hist > SAFE_THRESHOLD

    valleys=[]
    curr=[]

    for i,v in enumerate(valid):

        if v:
            curr.append(i)
        elif curr:
            valleys.append(curr)
            curr=[]

    if curr:
        valleys.append(curr)

    best=None
    best_width=0

    for val in valleys:

        if len(val)>best_width and len(val)>MIN_VALLEY_WIDTH:
            best_width=len(val)
            best=val[len(val)//2]

    return angles[best] if best is not None else None


# ================= MAVLINK =================

def send_body_velocity(vx,vy,vz):

    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000111111000111,
        0,0,0,
        vx,vy,vz,
        0,0,0,
        0,0
    )


# ================= CONNECT =================

client = airsim.MultirotorClient(ip=WINDOWS_IP)
client.confirmConnection()

master = mavutil.mavlink_connection("/dev/ttyUSB0",baud=57600)
master.wait_heartbeat()

model = TRT(ENGINE_PATH)

print("System Ready")


# ================= OFFBOARD INIT =================

print("Initializing Offboard")

for _ in range(100):
    send_body_velocity(0,0,0)
    time.sleep(0.05)


master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_DO_SET_MODE,
    0,
    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
    6,
    0,0,0,0,0,0
)

time.sleep(1)


master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0,
    1,
    0,0,0,0,0,0,0
)

time.sleep(2)


# ================= TAKEOFF =================

print("Takeoff")

for _ in range(25):
    send_body_velocity(0,0,CLIMB_SPEED)
    time.sleep(0.1)


# ================= NAVIGATION =================

traj_x=[]
traj_y=[]

x=y=0
prev_steer=0

print("Navigation starting")

while True:

    depth, raw_depth, rgb = get_depth_map()

    hist,angles = compute_histogram(depth)

    steer = find_best_sector(hist,angles)

    center = hist[NUM_SECTORS//2]

    if center < 0.35:

        print("STOP obstacle")

        send_body_velocity(0,0,0)
        time.sleep(0.1)
        continue


    if center < 0.5:

        left=np.mean(hist[60:80])
        right=np.mean(hist[10:30])

        if left>right:
            send_body_velocity(0,0.4,0)
        else:
            send_body_velocity(0,-0.4,0)

        time.sleep(DT)
        continue


    if steer is None:
        steer=0


    steer=0.8*prev_steer+0.2*steer
    prev_steer=steer


    if center<0.55:
        speed=MIN_SPEED
    elif center<0.7:
        speed=1
    else:
        speed=BASE_SPEED


    turn_penalty=abs(steer)/radians(45)

    speed*=(1-0.6*turn_penalty)


    vx=cos(steer)*speed
    vy=sin(steer)*speed


    send_body_velocity(vx,vy,0)


    x+=vx*DT
    y+=vy*DT

    traj_x.append(x)
    traj_y.append(y)


    print("Front:",round(center,2),
          "Speed:",round(speed,2),
          "Steer:",round(degrees(steer),1))


    rgb_vis=cv2.resize(rgb,(DISPLAY_W,DISPLAY_H))

    raw_vis=cv2.resize((raw_depth*255).astype(np.uint8),(DISPLAY_W,DISPLAY_H))

    depth_color=cv2.applyColorMap(raw_vis,cv2.COLORMAP_PLASMA)

    cv2.imshow("AirSim RGB Image",rgb_vis)
    cv2.imshow("MiDaS Depth Map",depth_color)


    if cv2.waitKey(1)==ord('q'):
        break

    if sqrt(x*x+y*y)>MAX_DISTANCE:
        break

    time.sleep(DT)


# ================= LAND =================

print("Landing")

master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_NAV_LAND,
    0,0,0,0,0,0,0,0
)

cv2.destroyAllWindows()

plt.plot(traj_x,traj_y)
plt.title("Drone trajectory")
plt.grid()
plt.show()