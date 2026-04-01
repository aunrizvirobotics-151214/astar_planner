#!/usr/bin/env python3
"""
A* Navigation Node — ROS 2 Humble
===================================
Subscribes to:
  /map            (nav_msgs/OccupancyGrid)    — 2-D occupancy map
  /goal_pose      (geometry_msgs/PoseStamped) — navigation goal (RViz2 "2D Nav Goal")

Publishes:
  /astar/path     (nav_msgs/Path)             — planned path
  /astar/visited  (nav_msgs/OccupancyGrid)    — debug: A* expanded cells
  /cmd_vel        (geometry_msgs/Twist)        — velocity commands

TF:
  Reads map → base_link to determine current robot pose.
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos  import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import rclpy.time
from scipy.ndimage import binary_dilation, generate_binary_structure

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg      import OccupancyGrid, Path
from std_msgs.msg      import Header
import tf2_ros
import tf_transformations

from astar_navigation.planner import (
    plan_astar_4,
    plan_astar_8,
    nearest_free_cell,
)


# ──────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────

def wrap_to_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def quat_to_yaw(q):
    """geometry_msgs Quaternion → yaw (rad)."""
    return tf_transformations.euler_from_quaternion(
        [q.x, q.y, q.z, q.w])[2]


def yaw_to_quat_msg(yaw):
    from geometry_msgs.msg import Quaternion
    arr = tf_transformations.quaternion_from_euler(0.0, 0.0, yaw)
    return Quaternion(x=arr[0], y=arr[1], z=arr[2], w=arr[3])


#  Map coordinate helpers
#
#  OccupancyGrid layout (ROS convention):
#    data[row * W + col]
#    row 0  = world y = origin.y          (bottom of the map)
#    row H-1 = world y = origin.y + H*res (top)
#    col 0  = world x = origin.x
#
#  In numpy array (after reshape to [H, W]):
#    increasing row → increasing world y
#    increasing col → increasing world x
#
#  world_to_cell / cell_to_world are consistent with this convention.


def world_to_cell(wx, wy, origin_x, origin_y, resolution):
    """World (x, y) → grid (row, col).  May be out of bounds."""
    col = int(round((wx - origin_x) / resolution))
    row = int(round((wy - origin_y) / resolution))
    return row, col


def cell_to_world(row, col, origin_x, origin_y, resolution):
    """Grid (row, col) → world (x, y) — cell centre."""
    x = col * resolution + origin_x
    y = row * resolution + origin_y
    return x, y


def in_bounds(row, col, H, W):
    return 0 <= row < H and 0 <= col < W


# ──────────────────────────────────────────────────────
#  Path smoother (Ramer–Douglas–Peucker)
# ──────────────────────────────────────────────────────

def _rdp(points, eps):
    """Simplify a polyline with RDP.  points: list of (x, y)."""
    if len(points) < 3:
        return points
    start, end = np.array(points[0]), np.array(points[-1])
    seg  = end - start
    seg_len = np.linalg.norm(seg)
    if seg_len < 1e-9:
        dists = [np.linalg.norm(np.array(p) - start) for p in points[1:-1]]
    else:
        unit = seg / seg_len
        dists = [abs(np.cross(unit, np.array(p) - start))
                 for p in points[1:-1]]
    idx = int(np.argmax(dists)) + 1
    if dists[idx - 1] > eps:
        left  = _rdp(points[:idx + 1], eps)
        right = _rdp(points[idx:], eps)
        return left[:-1] + right
    return [points[0], points[-1]]


def smooth_path(path_world, resolution):
    """
    Simplify a list of (x, y) waypoints using RDP with eps = 1.5 cells.
    Keeps start and goal exact.
    """
    eps = 1.5 * resolution
    return _rdp(path_world, eps)


# ──────────────────────────────────────────────────────
#  ROS 2 Node
# ──────────────────────────────────────────────────────

class AStarNode(Node):

    def __init__(self):
        super().__init__('astar_node')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('eight_connected',    True)
        self.declare_parameter('inflation_radius',   0.15)   # metres
        self.declare_parameter('control_frequency',  10.0)   # Hz
        self.declare_parameter('lookahead_distance', 0.5)    # metres
        self.declare_parameter('max_linear_vel',     0.2)    # m/s
        self.declare_parameter('max_angular_vel',    1.0)    # rad/s
        self.declare_parameter('kp_angular',         1.5)    # P-gain heading
        self.declare_parameter('goal_tolerance',     0.20)   # metres
        self.declare_parameter('map_frame',         'map')
        self.declare_parameter('base_frame',        'base_link')
        self.declare_parameter('cmd_vel_topic',     '/cmd_vel')

        p = lambda n: self.get_parameter(n).value
        self._eight_connected  = p('eight_connected')
        self._infl_radius      = p('inflation_radius')
        self._ctrl_freq        = p('control_frequency')
        self._lookahead        = p('lookahead_distance')
        self._max_lin          = p('max_linear_vel')
        self._max_ang          = p('max_angular_vel')
        self._kp               = p('kp_angular')
        self._goal_tol         = p('goal_tolerance')
        self._map_frame        = p('map_frame')
        self._base_frame       = p('base_frame')
        self._cmd_topic        = p('cmd_vel_topic')

        # ── State ──────────────────────────────────────────────────────────
        self._free_map    = None   # (H, W) float: 1.0=free 0.0=obstacle
        self._map_res     = None   # metres/cell
        self._map_origin  = None   # [origin_x, origin_y] world coords
        self._map_shape   = None   # (H, W)
        self._map_info    = None   # OccupancyGrid.info (for publishing)

        self._path_world  = []     # list of (x, y) waypoints after smoothing
        self._path_idx    = 0      # next lookahead search start index
        self._goal_reached = True  # True = not navigating

        # ── QoS: map is latched (Transient Local) ─────────────────────────
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ── Subscriptions ──────────────────────────────────────────────────
        self.sub_map  = self.create_subscription(
            OccupancyGrid, '/map', self._map_callback, map_qos)
        self.sub_goal = self.create_subscription(
            PoseStamped, '/goal_pose', self._goal_callback, 10)

        # ── Publishers ─────────────────────────────────────────────────────
        self.pub_path    = self.create_publisher(Path,          '/astar/path',    10)
        self.pub_visited = self.create_publisher(OccupancyGrid, '/astar/visited', map_qos)
        self.pub_cmd     = self.create_publisher(Twist, self._cmd_topic, 10)

        # ── TF ─────────────────────────────────────────────────────────────
        self._tf_buf = tf2_ros.Buffer()
        self._tf_lst = tf2_ros.TransformListener(self._tf_buf, self)

        # ── Control timer ─────────────────────────────────────────────────
        dt = 1.0 / self._ctrl_freq
        self.create_timer(dt, self._control_callback)

        self.get_logger().info('AStarNode started — waiting for /map and /goal_pose …')

    # ──────────────────────────────────────────────────────────────────────
    #  Map callback
    # ──────────────────────────────────────────────────────────────────────

    def _map_callback(self, msg: OccupancyGrid):
        info          = msg.info
        H, W          = info.height, info.width
        self._map_res    = info.resolution
        self._map_origin = [info.origin.position.x, info.origin.position.y]
        self._map_shape  = (H, W)
        self._map_info   = info

        # OccupancyGrid: -1=unknown, 0=free, 100=occupied
        raw = np.array(msg.data, dtype=np.int16).reshape(H, W)

        # Float freespace array: 1.0=free, 0.0=obstacle/unknown
        free = np.zeros((H, W), dtype=np.float32)
        free[raw == 0] = 1.0

        # Inflate obstacles by inflation_radius
        infl_cells = max(1, int(math.ceil(self._infl_radius / self._map_res)))
        struct     = generate_binary_structure(2, 2)     # 8-connected disk
        obstacle   = free < 0.9                          # everything non-free
        inflated   = binary_dilation(obstacle, structure=struct,
                                     iterations=infl_cells)
        free[inflated] = 0.0

        self._free_map = free
        self.get_logger().info(
            f'Map received ({W}×{H}, res={self._map_res:.3f} m/cell, '
            f'inflation={infl_cells} cells).')

    # ──────────────────────────────────────────────────────────────────────
    #  Goal callback — plan and start following
    # ──────────────────────────────────────────────────────────────────────

    def _goal_callback(self, msg: PoseStamped):
        if self._free_map is None:
            self.get_logger().warn('Goal received but map not yet available — ignoring.')
            return

        # ── Get robot pose in map frame ────────────────────────────────────
        try:
            tf = self._tf_buf.lookup_transform(
                self._map_frame, self._base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return

        rx = tf.transform.translation.x
        ry = tf.transform.translation.y

        # ── Goal world coords ─────────────────────────────────────────────
        gx = msg.pose.position.x
        gy = msg.pose.position.y

        self.get_logger().info(
            f'Planning: robot ({rx:.2f}, {ry:.2f}) → goal ({gx:.2f}, {gy:.2f})')

        # ── World → cell ───────────────────────────────────────────────────
        H, W = self._map_shape
        ox, oy = self._map_origin
        res    = self._map_res

        r_start, c_start = world_to_cell(rx, ry, ox, oy, res)
        r_goal,  c_goal  = world_to_cell(gx, gy, ox, oy, res)

        if not in_bounds(r_start, c_start, H, W):
            self.get_logger().warn('Robot start cell is outside the map — ignoring goal.')
            return
        if not in_bounds(r_goal, c_goal, H, W):
            self.get_logger().warn('Goal cell is outside the map — ignoring.')
            return

        # Snap start/goal to nearest free cell if in an inflated obstacle
        n_start = nearest_free_cell([r_start, c_start], self._free_map)
        n_goal  = nearest_free_cell([r_goal,  c_goal],  self._free_map)

        # ── Run A* ────────────────────────────────────────────────────────
        plan_fn = plan_astar_8 if self._eight_connected else plan_astar_4
        path_cells, visited = plan_fn(n_start, n_goal, self._free_map)

        if not path_cells:
            self.get_logger().warn('A*: no path found to goal.')
            self._publish_visited(visited)
            return

        self.get_logger().info(f'A*: path found ({len(path_cells)} cells).')

        # ── Cell path → world coords ───────────────────────────────────────
        path_world = [
            cell_to_world(int(c[0]), int(c[1]), ox, oy, res)
            for c in path_cells
        ]

        # ── Smooth ────────────────────────────────────────────────────────
        path_world = smooth_path(path_world, res)
        self.get_logger().info(f'Smoothed path: {len(path_world)} waypoints.')

        # ── Store and start following ─────────────────────────────────────
        self._path_world   = path_world
        self._path_idx     = 0
        self._goal_reached = False

        # ── Publish ───────────────────────────────────────────────────────
        self._publish_path(msg.header.stamp)
        self._publish_visited(visited)

    # ──────────────────────────────────────────────────────────────────────
    #  Control timer — pure pursuit
    # ──────────────────────────────────────────────────────────────────────

    def _control_callback(self):
        if self._goal_reached or not self._path_world:
            return

        # ── Get robot pose ─────────────────────────────────────────────────
        try:
            tf = self._tf_buf.lookup_transform(
                self._map_frame, self._base_frame,
                rclpy.time.Time())
        except Exception:
            return

        rx  = tf.transform.translation.x
        ry  = tf.transform.translation.y
        rth = quat_to_yaw(tf.transform.rotation)

        path = self._path_world
        N    = len(path)

        # ── Check goal reached ─────────────────────────────────────────────
        gx, gy = path[-1]
        if math.hypot(gx - rx, gy - ry) < self._goal_tol:
            self._stop()
            self._goal_reached = True
            self.get_logger().info('Goal reached!')
            return

        # ── Advance path_idx past captured waypoints ──────────────────────
        # A waypoint is "captured" once the robot is within capture_radius of
        # it.  Advancing here guarantees the lookahead search always starts
        # from a waypoint that is still AHEAD of the robot, preventing the
        # bug where a behind-waypoint satisfies the >= lookahead condition
        # and causes the robot to spin trying to return to it.
        capture_r = self._lookahead * 0.5   # must stay < lookahead
        while self._path_idx < N - 1:
            px, py = path[self._path_idx]
            if math.hypot(px - rx, py - ry) < capture_r:
                self._path_idx += 1
            else:
                break

        # ── Find lookahead point ───────────────────────────────────────────
        # Search forward from the first uncaptured waypoint for the nearest
        # point that is >= lookahead_distance from the robot.
        # Because we always start from an uncaptured (ahead) waypoint, this
        # point is guaranteed to be in front of the robot.
        la_i = self._path_idx   # fallback: steer toward next target directly
        for i in range(self._path_idx, N):
            px, py = path[i]
            if math.hypot(px - rx, py - ry) >= self._lookahead:
                la_i = i
                break

        lx, ly        = path[la_i]
        angle_to_la   = math.atan2(ly - ry, lx - rx)
        heading_err   = wrap_to_pi(angle_to_la - rth)

        # ── Compute velocities ─────────────────────────────────────────────
        # Reduce speed proportional to turn sharpness
        speed_factor = max(0.15, 1.0 - abs(heading_err) / math.pi * 0.85)
        lin_vel = self._max_lin * speed_factor
        ang_vel = np.clip(self._kp * heading_err, -self._max_ang, self._max_ang)

        cmd             = Twist()
        cmd.linear.x    = lin_vel
        cmd.angular.z   = float(ang_vel)
        self.pub_cmd.publish(cmd)

    def _stop(self):
        self.pub_cmd.publish(Twist())

    # ──────────────────────────────────────────────────────────────────────
    #  Publishers
    # ──────────────────────────────────────────────────────────────────────

    def _publish_path(self, stamp):
        from geometry_msgs.msg import PoseStamped as PS
        msg        = Path()
        msg.header = Header(frame_id=self._map_frame, stamp=stamp)
        poses      = []
        for i, (x, y) in enumerate(self._path_world):
            ps                  = PS()
            ps.header           = msg.header
            ps.pose.position.x  = x
            ps.pose.position.y  = y
            ps.pose.position.z  = 0.0
            # Yaw: direction to next waypoint (last waypoint keeps previous)
            if i + 1 < len(self._path_world):
                nx, ny = self._path_world[i + 1]
                yaw    = math.atan2(ny - y, nx - x)
            else:
                yaw = 0.0
            ps.pose.orientation = yaw_to_quat_msg(yaw)
            poses.append(ps)
        msg.poses = poses
        self.pub_path.publish(msg)

    def _publish_visited(self, visited: np.ndarray):
        """Publish the A* expanded-cell grid as an OccupancyGrid for RViz2."""
        if self._map_info is None:
            return
        # visited: 0.0=not expanded, 1.0=expanded → scale to [0, 50]
        scaled = (visited * 50).astype(np.int8)
        msg        = OccupancyGrid()
        msg.header = Header(
            frame_id=self._map_frame,
            stamp=self.get_clock().now().to_msg())
        msg.info   = self._map_info
        msg.data   = scaled.flatten().tolist()
        self.pub_visited.publish(msg)


# ──────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = AStarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
