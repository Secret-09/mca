"""
Parallel Traffic Simulation with Time-Dependent Optimization
============================================================
Uses multiprocessing.Pool to bypass Python GIL and achieve real speedup.
Sequential vs Parallel comparison shows drastic performance difference.
"""

import os
import json
import math
import random
import time
import threading
import multiprocessing
from multiprocessing import Pool
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
NUM_ROADS = 4
ROAD_NAMES = ['NORTH', 'SOUTH', 'EAST', 'WEST']
BASE_GREEN_TIME = 10.0
MIN_GREEN_TIME  = 3.0
MAX_GREEN_TIME  = 25.0
YELLOW_TIME     = 2.0

# Uneven road loads — North heavy, East very heavy, South/West light
ROAD_WEIGHTS = [1.0, 0.3, 1.8, 0.2]

# ─── VEHICLE MODEL ────────────────────────────────────────────────────────────
def make_vehicle(vid, road_id, spawn_time):
    speed = random.uniform(40, 80)
    road_dist = random.uniform(200, 600)
    return {
        'id': vid,
        'road_id': road_id,
        'spawn_time': spawn_time,
        'distance': road_dist,
        'speed': speed,
        'waiting': False,
        'wait_time': 0.0,
        'passed': False,
        'pass_time': None,
        'acceleration': random.uniform(-2, 2),
        'mass': random.uniform(1000, 3000),
    }


def _compute_chunk(args):
    """
    Top-level picklable function for multiprocessing.
    Each process handles a chunk of vehicles independently.
    """
    chunk, dt, signal_state = args
    results = []
    for v in chunk:
        results.append(_update_one_heavy(v, dt, signal_state))
    return results


def _update_one_heavy(v, dt, signal_state):
    """
    Heavy physics update per vehicle — includes expensive math
    to simulate real braking/IDM/emission models.
    This is what we parallelize across processes.
    """
    if v['passed']:
        return v

    nv = dict(v)
    road_pair = 0 if nv['road_id'] in (0, 1) else 1
    sig = signal_state.get(road_pair, 'red')
    STOP_DISTANCE = 15.0

    # ── Expensive per-vehicle computation ──
    # Simulates: IDM model, fuel consumption, braking curves, emission estimation
    # This heavy loop is what creates the sequential vs parallel time difference
    result = 0.0
    dist = nv['distance']
    spd = nv['speed']
    mass = nv['mass']
    for i in range(4000):
        result += math.sin(dist * 0.01 + i * 0.001) * math.cos(spd * 0.01)
        result += math.sqrt(abs(nv['acceleration']) + 0.001)
        result += math.log(mass + 1.0) * 0.0001
        if i % 10 == 0:
            result -= math.exp(-abs(result) * 0.001) * 0.0001

    # Braking distance formula
    v_ms = nv['speed'] / 3.6
    braking_dist = (v_ms * v_ms) / (2.0 * 0.7 * 9.81)

    if nv['waiting']:
        if sig == 'green':
            nv['waiting'] = False
            nv['acceleration'] = 2.0
        else:
            nv['wait_time'] += dt
            return nv
    else:
        if nv['distance'] <= (STOP_DISTANCE + braking_dist) and sig != 'green':
            decel = min(6.0, nv['speed'] / max(dt, 0.001))
            nv['speed'] = max(0.0, nv['speed'] - decel * dt * 3.6)
            if nv['distance'] <= STOP_DISTANCE and nv['speed'] < 2.0:
                nv['speed'] = 0.0
                nv['waiting'] = True
                nv['acceleration'] = 0.0
                return nv
        else:
            nv['speed'] = max(5.0, min(100.0, nv['speed'] + (60.0 - nv['speed']) * 0.1 * dt))

        nv['distance'] -= nv['speed'] / 3.6 * dt
        if nv['distance'] <= 0:
            nv['passed'] = True
            nv['distance'] = 0.0

    return nv


def _update_one_light(v, dt, signal_state):
    """Light version for live real-time display (no heavy computation)."""
    if v['passed']:
        return v

    nv = dict(v)
    road_pair = 0 if nv['road_id'] in (0, 1) else 1
    sig = signal_state.get(road_pair, 'red')
    STOP_DISTANCE = 15.0

    v_ms = nv['speed'] / 3.6
    braking_dist = (v_ms * v_ms) / (2.0 * 0.7 * 9.81)

    if nv['waiting']:
        if sig == 'green':
            nv['waiting'] = False
            nv['acceleration'] = 2.0
        else:
            nv['wait_time'] += dt
            return nv
    else:
        if nv['distance'] <= (STOP_DISTANCE + braking_dist) and sig != 'green':
            nv['speed'] = max(0.0, nv['speed'] - 8.0 * dt * 3.6)
            if nv['distance'] <= STOP_DISTANCE and nv['speed'] < 2.0:
                nv['speed'] = 0.0
                nv['waiting'] = True
                return nv
        else:
            nv['speed'] = max(5.0, min(80.0, nv['speed'] + (60.0 - nv['speed']) * 0.1 * dt))

        nv['distance'] -= nv['speed'] / 3.6 * dt
        if nv['distance'] <= 0:
            nv['passed'] = True
            nv['distance'] = 0.0

    return nv


# ─── SIGNAL OPTIMIZER ─────────────────────────────────────────────────────────
class SignalOptimizer:
    def __init__(self):
        self.road_counts = [0] * NUM_ROADS

    def update_counts(self, vehicles):
        counts = [0] * NUM_ROADS
        for v in vehicles:
            if not v['passed']:
                counts[v['road_id']] += 1
        self.road_counts = counts

    def compute_green_times(self, mode='adaptive'):
        ns = self.road_counts[0] + self.road_counts[1]
        ew = self.road_counts[2] + self.road_counts[3]
        total = ns + ew

        if mode == 'static' or total == 0:
            return [BASE_GREEN_TIME, BASE_GREEN_TIME]

        total_budget = BASE_GREEN_TIME * 2
        min_g = MIN_GREEN_TIME
        remaining = total_budget - 2 * min_g
        ns_time = min_g + remaining * (ns / total)
        ew_time = min_g + remaining * (ew / total)

        return [
            round(max(min_g, min(MAX_GREEN_TIME, ns_time)), 2),
            round(max(min_g, min(MAX_GREEN_TIME, ew_time)), 2),
        ]


# ─── SEQUENTIAL SIMULATION ────────────────────────────────────────────────────
def run_sequential(vehicles, total_sim_time=30.0, signal_mode='adaptive'):
    """Sequential: one process, one loop, vehicles updated one by one."""
    optimizer = SignalOptimizer()
    current_pair = 0
    signal_phase = 'green'
    signal_timer = 0.0
    green_times = [BASE_GREEN_TIME, BASE_GREEN_TIME]

    dt = 0.1
    steps = int(total_sim_time / dt)

    total_wait = 0.0
    wasted_green = 0.0
    compute_times = []

    for step in range(steps):
        signal_timer += dt
        if signal_phase == 'green':
            active_roads = [0, 1] if current_pair == 0 else [2, 3]
            waiting_on_green = sum(1 for v in vehicles
                                   if v['road_id'] in active_roads and not v['passed'])
            if waiting_on_green == 0:
                wasted_green += dt
            if signal_timer >= green_times[current_pair]:
                signal_phase = 'yellow'
                signal_timer = 0.0
        else:
            if signal_timer >= YELLOW_TIME:
                current_pair = 1 - current_pair
                signal_phase = 'green'
                signal_timer = 0.0
                optimizer.update_counts(vehicles)
                green_times = optimizer.compute_green_times(signal_mode)

        signal_state = {current_pair: signal_phase, 1 - current_pair: 'red'}

        # SEQUENTIAL: one vehicle at a time, single thread
        t0 = time.perf_counter()
        new_vehicles = []
        for v in vehicles:
            new_vehicles.append(_update_one_heavy(v, dt, signal_state))
        t1 = time.perf_counter()
        compute_times.append(t1 - t0)
        vehicles = new_vehicles

        for v in vehicles:
            if v['passed'] and v['wait_time'] > 0:
                total_wait += v['wait_time']
                v['wait_time'] = 0

    passed_count = sum(1 for v in vehicles if v['passed'])
    avg_wait = total_wait / max(passed_count, 1)

    return {
        'passed': passed_count,
        'avg_wait': round(avg_wait, 3),
        'wasted_green': round(wasted_green, 2),
        'total_compute_ms': round(sum(compute_times) * 1000, 2),
        'avg_step_ms': round(sum(compute_times) / max(len(compute_times), 1) * 1000, 4),
    }


# ─── PARALLEL SIMULATION ──────────────────────────────────────────────────────
def run_parallel(vehicles, total_sim_time=30.0, signal_mode='adaptive', num_processes=4):
    """
    Parallel: multiprocessing.Pool — true CPU parallelism, bypasses Python GIL.
    Vehicles partitioned into chunks; each process handles one chunk independently.
    """
    optimizer = SignalOptimizer()
    current_pair = 0
    signal_phase = 'green'
    signal_timer = 0.0
    green_times = [BASE_GREEN_TIME, BASE_GREEN_TIME]

    dt = 0.1
    steps = int(total_sim_time / dt)

    total_wait = 0.0
    wasted_green = 0.0
    compute_times = []

    pool = Pool(processes=num_processes)
    try:
        for step in range(steps):
            signal_timer += dt
            if signal_phase == 'green':
                active_roads = [0, 1] if current_pair == 0 else [2, 3]
                waiting_on_green = sum(1 for v in vehicles
                                       if v['road_id'] in active_roads and not v['passed'])
                if waiting_on_green == 0:
                    wasted_green += dt
                if signal_timer >= green_times[current_pair]:
                    signal_phase = 'yellow'
                    signal_timer = 0.0
            else:
                if signal_timer >= YELLOW_TIME:
                    current_pair = 1 - current_pair
                    signal_phase = 'green'
                    signal_timer = 0.0
                    optimizer.update_counts(vehicles)
                    green_times = optimizer.compute_green_times(signal_mode)

            signal_state = {current_pair: signal_phase, 1 - current_pair: 'red'}

            # PARALLEL: split vehicles into N chunks, pool.map runs them concurrently
            n = len(vehicles)
            chunk_size = max(1, n // num_processes)
            chunks = []
            for i in range(0, n, chunk_size):
                chunks.append(vehicles[i:i + chunk_size])

            args_list = [(chunk, dt, signal_state) for chunk in chunks]

            t0 = time.perf_counter()
            chunk_results = pool.map(_compute_chunk, args_list)
            t1 = time.perf_counter()
            compute_times.append(t1 - t0)

            vehicles = [v for chunk_result in chunk_results for v in chunk_result]

            for v in vehicles:
                if v['passed'] and v['wait_time'] > 0:
                    total_wait += v['wait_time']
                    v['wait_time'] = 0
    finally:
        pool.close()
        pool.join()

    passed_count = sum(1 for v in vehicles if v['passed'])
    avg_wait = total_wait / max(passed_count, 1)

    return {
        'passed': passed_count,
        'avg_wait': round(avg_wait, 3),
        'wasted_green': round(wasted_green, 2),
        'total_compute_ms': round(sum(compute_times) * 1000, 2),
        'avg_step_ms': round(sum(compute_times) / max(len(compute_times), 1) * 1000, 4),
    }


# ─── VEHICLE GENERATOR ────────────────────────────────────────────────────────
def generate_vehicles(n):
    total_w = sum(ROAD_WEIGHTS)
    vehicles = []
    for i in range(n):
        r = random.random() * total_w
        road_id = len(ROAD_WEIGHTS) - 1
        for j, w in enumerate(ROAD_WEIGHTS):
            r -= w
            if r <= 0:
                road_id = j
                break
        vehicles.append(make_vehicle(i, road_id, random.uniform(0, 30)))
    return vehicles


# ─── BENCHMARK ────────────────────────────────────────────────────────────────
def run_benchmark(num_vehicles, signal_mode, process_counts):
    results = {}
    base_vehicles = generate_vehicles(num_vehicles)

    print(f"[Benchmark] Sequential: {num_vehicles} vehicles, 30s sim...")
    t_start = time.perf_counter()
    seq_stats = run_sequential([dict(v) for v in base_vehicles], 30.0, signal_mode)
    seq_wall = time.perf_counter() - t_start
    seq_stats['wall_time_s'] = round(seq_wall, 3)
    results['sequential'] = seq_stats
    print(f"  Done: {seq_wall:.2f}s")

    results['parallel'] = {}
    for pc in process_counts:
        print(f"[Benchmark] Parallel {pc} processes...")
        t_start = time.perf_counter()
        par_stats = run_parallel([dict(v) for v in base_vehicles], 30.0, signal_mode, pc)
        par_wall = time.perf_counter() - t_start
        par_stats['wall_time_s'] = round(par_wall, 3)
        par_stats['speedup'] = round(seq_wall / par_wall, 2)
        par_stats['threads'] = pc
        results['parallel'][str(pc)] = par_stats
        print(f"  Done: {par_wall:.2f}s, speedup={par_stats['speedup']}x")

    return results


# ─── LIVE SIMULATION ──────────────────────────────────────────────────────────
class LiveSimulation:
    def __init__(self):
        self.lock = threading.Lock()
        self.vehicles = []
        self.signal = {'pair': 0, 'phase': 'green', 'timer': 0.0}
        self.green_times = [BASE_GREEN_TIME, BASE_GREEN_TIME]
        self.road_counts = [0] * NUM_ROADS
        self.mode = 'adaptive'
        self.running = False
        self.sim_time = 0.0
        self.passed = 0
        self.total_wait = 0.0
        self.optimizer = SignalOptimizer()
        self.thread = None
        self.spawn_timer = 0.0
        self.next_id = 0
        self.wasted = 0.0

    def start(self, mode='adaptive'):
        self.mode = mode
        self.running = True
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False

    def reset(self):
        with self.lock:
            self.running = False
            self.vehicles = []
            self.sim_time = 0.0
            self.passed = 0
            self.total_wait = 0.0
            self.signal = {'pair': 0, 'phase': 'green', 'timer': 0.0}
            self.green_times = [BASE_GREEN_TIME, BASE_GREEN_TIME]
            self.road_counts = [0] * NUM_ROADS
            self.spawn_timer = 0.0
            self.next_id = 0
            self.wasted = 0.0

    def _spawn(self, dt):
        self.spawn_timer += dt
        while self.spawn_timer >= 0.9:
            self.spawn_timer -= 0.9
            tw = sum(ROAD_WEIGHTS)
            r = random.random() * tw
            road_id = len(ROAD_WEIGHTS) - 1
            for j, w in enumerate(ROAD_WEIGHTS):
                r -= w
                if r <= 0:
                    road_id = j
                    break
            self.vehicles.append(make_vehicle(self.next_id, road_id, self.sim_time))
            self.next_id += 1

    def _loop(self):
        dt = 0.05
        while self.running:
            t0 = time.perf_counter()
            with self.lock:
                self.sim_time += dt
                self._spawn(dt)

                sig = self.signal
                sig['timer'] += dt
                pair = sig['pair']

                if sig['phase'] == 'green':
                    active = [0, 1] if pair == 0 else [2, 3]
                    if all(self.road_counts[r] == 0 for r in active):
                        self.wasted += dt
                    if sig['timer'] >= self.green_times[pair]:
                        sig['phase'] = 'yellow'
                        sig['timer'] = 0.0
                else:
                    if sig['timer'] >= YELLOW_TIME:
                        sig['pair'] = 1 - pair
                        sig['phase'] = 'green'
                        sig['timer'] = 0.0
                        self.optimizer.update_counts(self.vehicles)
                        self.green_times = self.optimizer.compute_green_times(self.mode)

                signal_state = {sig['pair']: sig['phase'], 1 - sig['pair']: 'red'}

                updated = []
                for v in self.vehicles:
                    nv = _update_one_light(v, dt, signal_state)
                    if nv['passed'] and nv['wait_time'] > 0:
                        self.passed += 1
                        self.total_wait += nv['wait_time']
                        nv['wait_time'] = 0
                    updated.append(nv)

                self.vehicles = [v for v in updated
                                 if not (v['passed'] and v['distance'] < -50)]

                counts = [0] * NUM_ROADS
                for v in self.vehicles:
                    if not v['passed']:
                        counts[v['road_id']] += 1
                self.road_counts = counts

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))

    def get_state(self):
        with self.lock:
            vehicles_snapshot = [
                {
                    'id': v['id'],
                    'road_id': v['road_id'],
                    'distance': round(v['distance'], 1),
                    'speed': round(v['speed'], 1),
                    'waiting': v['waiting'],
                    'passed': v['passed'],
                }
                for v in self.vehicles[-120:]
            ]
            return {
                'sim_time': round(self.sim_time, 1),
                'signal': {
                    'pair': self.signal['pair'],
                    'phase': self.signal['phase'],
                    'timer': round(self.signal['timer'], 1),
                    'green_times': self.green_times,
                },
                'road_counts': self.road_counts,
                'passed': self.passed,
                'avg_wait': round(self.total_wait / max(self.passed, 1), 2),
                'queued': sum(1 for v in self.vehicles if v['waiting']),
                'active': len(self.vehicles),
                'wasted': round(self.wasted, 1),
                'vehicles': vehicles_snapshot,
                'mode': self.mode,
            }


# ─── HTTP SERVER ──────────────────────────────────────────────────────────────
live_sim = LiveSimulation()
benchmark_results = {}
benchmark_running = False


def read_html():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(script_dir, 'index.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        return f.read()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        global benchmark_results, benchmark_running
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == '/':
            html = read_html().encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(html))
            self.end_headers()
            self.wfile.write(html)

        elif path == '/api/state':
            self.send_json(live_sim.get_state())

        elif path == '/api/start':
            mode = params.get('mode', ['adaptive'])[0]
            live_sim.start(mode)
            self.send_json({'ok': True, 'mode': mode})

        elif path == '/api/stop':
            live_sim.stop()
            self.send_json({'ok': True})

        elif path == '/api/reset':
            live_sim.reset()
            self.send_json({'ok': True})

        elif path == '/api/benchmark':
            if benchmark_running:
                self.send_json({'status': 'already_running'})
                return

            n = int(params.get('n', [150])[0])
            mode = params.get('mode', ['adaptive'])[0]
            proc_counts = [1, 2, 4]

            def run_bg():
                global benchmark_results, benchmark_running
                benchmark_running = True
                try:
                    benchmark_results = run_benchmark(n, mode, proc_counts)
                    benchmark_results['status'] = 'done'
                    benchmark_results['num_vehicles'] = n
                    benchmark_results['mode'] = mode
                except Exception as e:
                    import traceback
                    benchmark_results = {
                        'status': 'error',
                        'error': str(e),
                        'trace': traceback.format_exc()
                    }
                finally:
                    benchmark_running = False

            t = threading.Thread(target=run_bg, daemon=True)
            t.start()
            self.send_json({'status': 'started'})

        elif path == '/api/benchmark_result':
            if benchmark_running:
                self.send_json({'status': 'running'})
            elif benchmark_results:
                self.send_json({'status': 'done', 'data': benchmark_results})
            else:
                self.send_json({'status': 'idle'})

        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # IMPORTANT: use dynamic port for cloud
    PORT = int(os.environ.get("PORT", 8765))

    cpu_count = multiprocessing.cpu_count()
    print(f"\n{'='*55}")
    print(f"  Parallel Traffic Simulation Server")
    print(f"  Open in browser on port: {PORT}")
    print(f"  CPUs available: {cpu_count}")
    print(f"  Parallelism: multiprocessing.Pool (bypasses GIL)")
    print(f"{'='*55}\n")

    server = HTTPServer(("0.0.0.0", PORT), Handler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
