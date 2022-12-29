"""

Car model for Hybrid A* path planning

author: Zheng Zh (@Zhengzh)

"""

from math import cos, sin, tan, pi
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot

WB = 1.0  # rear to front wheel
W = 2.0  # width of car
R = W/2.0
LF = 3.3  # distance from rear to vehicle front end
LB = 1.0  # distance from rear to vehicle back end
MAX_STEER = 1.0  # [rad] maximum steering angle

BUBBLE_DIST = (LF - LB) / 2.0  # distance from rear to center of vehicle.
BUBBLE_R = np.hypot((LF + LB) / 2.0, W / 2.0)  # bubble radius

# vehicle rectangle vertices
VRX = [LF, LF, -LB, -LB, LF]
VRY = [W / 2, -W / 2, -W / 2, W / 2, W / 2]


def check_car_collision_dis(slist, occ_index, kd_tree):
    slist = slist.detach().numpy()
    # collisiondis_list = [0]*slist.shape[0]
    collisiondis_list = torch.zeros(slist.shape[0])
    nl = range(slist.shape[0])
    for n, s in zip(nl,slist):
        cx = s[0] + BUBBLE_DIST * cos(s[2])
        cy = s[1] + BUBBLE_DIST * sin(s[2])

        ids = kd_tree.query_ball_point([cx, cy], BUBBLE_R)
        if not ids:
            continue
        dis = rectangle_check_dis(s[0], s[1], s[2],
                              [occ_index[i][0] for i in ids], [occ_index[i][1] for i in ids])
        if dis:
            collisiondis_list[n] = dis
    return collisiondis_list  # no collision


def rectangle_check_dis(x, y, yaw, ox, oy):
    # transform obstacles to base link frame
    rot = Rot.from_euler('z', yaw).as_matrix()[0:2, 0:2]
    for iox, ioy in zip(ox, oy):
        tx = iox - x
        ty = ioy - y
        converted_xy = np.stack([tx, ty]).T @ rot
        rx, ry = converted_xy[0], converted_xy[1]

        if not (rx > LF or rx < -LB or ry > R or ry < -R):
            return False  # no collision
        else:
            if rx >= 0:
                clx = LF-rx
            else:
                clx = LB+rx

            if ry >= 0:
                cly = R-ry
            else:
                cly = R+ry
            return clx*cly
    # return True  # collision


def check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
    for i_x, i_y, i_yaw in zip(x_list, y_list, yaw_list):
        cx = i_x + BUBBLE_DIST * cos(i_yaw)
        cy = i_y + BUBBLE_DIST * sin(i_yaw)

        ids = kd_tree.query_ball_point([cx, cy], BUBBLE_R)

        if not ids:
            continue

        if not rectangle_check(i_x, i_y, i_yaw,
                               [ox[i] for i in ids], [oy[i] for i in ids]):
            return False  # collision

    return True  # no collision


def rectangle_check(x, y, yaw, ox, oy):
    # transform obstacles to base link frame
    rot = Rot.from_euler('z', yaw).as_matrix()[0:2, 0:2]
    for iox, ioy in zip(ox, oy):
        tx = iox - x
        ty = ioy - y
        converted_xy = np.stack([tx, ty]).T @ rot
        rx, ry = converted_xy[0], converted_xy[1]

        if not (rx > LF or rx < -LB or ry > R or ry < -R):
            return False  # no collision
        # else:
        #     if rx >= 0:
        #         clx = LF-rx
        #     else:
        #         clx = LB+rx

        #     if ry >= 0:
        #         cly = R-ry
        #     else:
        #         cly = R+ry
        #     return clx*cly
    return True  # collision


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """Plot arrow."""
    if not isinstance(x, float):
        for (i_x, i_y, i_yaw) in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw)
    else:
        plt.arrow(x, y, length * cos(yaw), length * sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width, alpha=0.4)


def plot_car(x, y, yaw):
    car_color = '-k'
    c, s = cos(yaw), sin(yaw)
    rot = Rot.from_euler('z', -yaw).as_matrix()[0:2, 0:2]
    car_outline_x, car_outline_y = [], []
    for rx, ry in zip(VRX, VRY):
        converted_xy = np.stack([rx, ry]).T @ rot
        car_outline_x.append(converted_xy[0]+x)
        car_outline_y.append(converted_xy[1]+y)

    arrow_x, arrow_y, arrow_yaw = c * 1.5 + x, s * 1.5 + y, yaw
    plot_arrow(arrow_x, arrow_y, arrow_yaw)

    plt.plot(car_outline_x, car_outline_y, car_color)


def pi_2_pi(angle):
    return (angle + pi) % (2 * pi) - pi

def pi_2_pi_pos(angle):
    return (angle + 2*pi) % (2 * pi)

def rad_2_degree(rad):
    return (pi_2_pi(rad)/pi)*180


def degree_2_rad(degree):
    return pi_2_pi((degree/180)*pi)


def steering_to_yawrate(steer, v, L=WB):
    return tan(steer)*v/L

def torch_yawrate_to_steering(yawrate,v,L=WB):
    return torch.atan2(yawrate*L,v)

def get_arc_length(array: np.ndarray) -> np.ndarray:
    displacement = np.cumsum(
        np.linalg.norm(np.diff(array[:, :2], axis=0), axis=1), axis=0
    )
    return np.pad(displacement, (1, 0), mode="constant")

def reduce_way_point(waypoint:np.ndarray, interval = 0.2)-> np.ndarray:
    ds = np.linalg.norm(np.diff(waypoint[...,:2],n=1,axis=0),axis=-1)
    ds = ds
    wp = waypoint[:-1,:2]
    w = [wp[0]]
    sl = 0.
    # ss = [sl]
    last_dw = 0
    for wp_t,ds_t in zip(wp,ds):
        if sl-last_dw>=interval:
            # ss.append(sl)
            w.append(wp_t)
            last_dw = sl
        sl += ds_t
    w = np.stack(w,axis=0)
    return w

def move(x, y, yaw, distance, steer, L=WB):
    x += distance * cos(yaw)
    y += distance * sin(yaw)
    yaw += pi_2_pi(distance * tan(steer) / L)  # distance/2

    return [x, y, yaw]


def main():
    x, y, yaw = 0., 0., 1.
    plt.axis('equal')
    plot_car(x, y, yaw)
    plt.show()


if __name__ == '__main__':
    main()
