import asyncio
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
import airsim
import cv2
import matplotlib.pyplot as plt
from math import radians, degrees, cos, sin, sqrt
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed

# ================= PARAMETERS =================
SAFE_THRESHOLD = 0.22
NUM_SECTORS = 91
MIN_VALLEY_WIDTH = 8
BASE_SPEED = 1          # Fixed speed
TARGET_ALT = 7.0          # Fixed altitude
MAX_DISTANCE = 15

# ================= MiDaS =================
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
])

sess = ort.InferenceSession(
    "midas-small.onnx",
    providers=['CPUExecutionProvider']
)

# ================= AirSim =================
client = airsim.MultirotorClient()
client.confirmConnection()

def get_depth_map():
    try:
        resp = client.simGetImages([
            airsim.ImageRequest("front_center",
                                airsim.ImageType.Scene,
                                False, False)
        ])[0]

        img = np.frombuffer(resp.image_data_uint8,
                            dtype=np.uint8).reshape(resp.height,
                                                    resp.width, 3)

        img = Image.fromarray(img)
        w, h = img.size

        if w > h:
            img = img.crop(((w-h)//2, 0, (w+h)//2, h))
        else:
            img = img.crop((0, (h-w)//2, w, (h+w)//2))

        inp = transform(img).unsqueeze(0).numpy()
        out = sess.run(None, {"input": inp})[0].squeeze()

        norm = (out - out.min()) / (out.max() - out.min())
        return 1.0 - norm

    except:
        return np.ones((224, 224))

# ================= Histogram =================
def compute_histogram(depth_map):
    h, w = depth_map.shape
    sector_w = w // NUM_SECTORS
    region = depth_map[int(0.35*h):int(0.75*h), :]

    hist = np.zeros(NUM_SECTORS)

    for i in range(NUM_SECTORS):
        col = region[:, i*sector_w:(i+1)*sector_w]
        v = col[col > 0]
        clearance = np.percentile(v, 25) if v.size > 0 else 1.0
        hist[i] = clearance

    angles = np.linspace(-np.radians(90),
                         np.radians(90),
                         NUM_SECTORS)

    return hist, angles

def find_best_sector(hist, angles):
    valid = hist > SAFE_THRESHOLD
    valleys, curr = [], []

    for i, ok in enumerate(valid):
        if ok:
            curr.append(i)
        elif curr:
            valleys.append(curr)
            curr = []

    if curr:
        valleys.append(curr)

    best = None
    widest = 0

    for val in valleys:
        if len(val) > widest:
            widest = len(val)
            best = val[len(val)//2]

    return angles[best] if best is not None else None

# ================= MAIN =================
async def run():

    print("🔗 Connecting to PX4...")
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14550")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("✅ Connected")
            break

    print("🛫 Arming...")
    await drone.action.arm()

    print("🛫 Taking off...")
    await drone.action.takeoff()
    await asyncio.sleep(8)

    print("🚀 Starting Offboard...")
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0, 0, 0, 0)
    )
    await drone.offboard.start()

    traj_x, traj_y = [], []
    x = y = 0.0
    dt = 0.1

    print("🚀 Navigation started")

    try:
        while True:

            depth = get_depth_map()
            hist, angles = compute_histogram(depth)
            steer = find_best_sector(hist, angles)

            if steer is None:
                vx = vy = 0.0
            else:
                vx = -cos(steer) * BASE_SPEED
                vy =  sin(steer) * BASE_SPEED

            # ===== ALTITUDE HOLD =====
            async for pos in drone.telemetry.position():
                current_alt = pos.relative_altitude_m
                break

            alt_error = TARGET_ALT - current_alt
            vz = -0.4 * alt_error
            vz = np.clip(vz, -0.5, 0.5)

            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(vx, vy, vz, 0)
            )

            # ===== Trajectory Update =====
            x += vx * dt
            y += vy * dt
            traj_x.append(x)
            traj_y.append(y)

            print(f"Alt:{current_alt:.2f} | "
                  f"Steer:{degrees(steer) if steer else 0:.1f}")

            depth_vis = (depth*255).astype(np.uint8)
            cv2.imshow("Depth", depth_vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if sqrt(x*x + y*y) > MAX_DISTANCE:
                break

            await asyncio.sleep(dt)

    except KeyboardInterrupt:
        pass

    print("🛬 Landing...")
    await drone.offboard.stop()
    await drone.action.land()
    await asyncio.sleep(8)

    print("✅ Landed")

    # ===== Trajectory Plot =====
    plt.figure(figsize=(8,6))
    plt.plot(traj_x, traj_y, 'b-', label="Drone Path")
    plt.plot(traj_x[-1], traj_y[-1], 'ro', label="Final Position")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Drone XY Trajectory")
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.show()

    cv2.destroyAllWindows()

asyncio.run(run())