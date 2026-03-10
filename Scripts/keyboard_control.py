import asyncio
import sys
import termios
import tty
import select

from mavsdk import System
from mavsdk.offboard import VelocityNedYaw

# ================= CONFIG =================
TARGET_ALT = 5.0        # meters
CLIMB_SPEED = -1.0      # m/s (negative = up in NED)
SPEED = 1.5             # horizontal speed (m/s)
YAW_RATE = 30.0         # deg/s
PORT = 14550
# =========================================


def get_key_nonblocking():
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


async def run():
    print("🔗 Connecting to PX4...")
    drone = System()
    await drone.connect(system_address=f"udpin://127.0.0.1:{PORT}")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("✅ Connected to PX4")
            break

    # ⚠️ DO NOT BLOCK on health in SITL
    print("🟢 Waiting for estimator to settle (5s)...")
    await asyncio.sleep(5)
    print("✅ Continuing (PX4 preflight OK)")

    print("🟢 Arming...")
    await drone.action.arm()
    print("✅ Armed")

    # ---------- OFFBOARD TAKEOFF ----------
    print("🟢 Preparing OFFBOARD...")
    await drone.offboard.set_velocity_ned(
        VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
    )

    await drone.offboard.start()
    print("✅ OFFBOARD started")

    print(f"🛫 Taking off to {TARGET_ALT} m")

    async for position in drone.telemetry.position():
        alt = position.relative_altitude_m
        print(f" Altitude: {alt:.2f} m")

        if alt >= TARGET_ALT:
            print("✅ Takeoff complete")
            break

        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(0.0, 0.0, CLIMB_SPEED, 0.0)
        )

        await asyncio.sleep(0.1)

    # Hover
    await drone.offboard.set_velocity_ned(
        VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
    )

    print("""
🎮 KEYBOARD CONTROL ACTIVE
-------------------------
W/S : North / South
A/D : West / East
R/F : Up / Down
Q/E : Yaw Left / Right
X   : Stop
CTRL+C : Land & Exit
""")

    # ---------- KEYBOARD CONTROL ----------
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    vx = vy = vz = yaw = 0.0

    try:
        while True:
            key = get_key_nonblocking()

            if key:
                print(f"KEY: {key}")

                if key == 'w':
                    vx = SPEED
                elif key == 's':
                    vx = -SPEED
                elif key == 'a':
                    vy = -SPEED
                elif key == 'd':
                    vy = SPEED
                elif key == 'r':
                    vz = -SPEED
                elif key == 'f':
                    vz = SPEED
                elif key == 'q':
                    yaw = -YAW_RATE
                elif key == 'e':
                    yaw = YAW_RATE
                elif key == 'x':
                    vx = vy = vz = yaw = 0.0

            # 🚨 CONTINUOUS OFFBOARD STREAM (CRITICAL)
            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(vx, vy, vz, yaw)
            )

            await asyncio.sleep(0.05)

    except KeyboardInterrupt:
        print("\n🛑 Landing & exiting...")

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
        )
        await drone.offboard.stop()
        await drone.action.land()


asyncio.run(run())
