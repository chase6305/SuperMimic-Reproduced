"""Render saved physical trajectories with Mesa llvmpipe, without CUDA allocation."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from supermimic.artifacts import sha256, write_json


def render_rollout(
    bundle: Path,
    trajectory: Path,
    output: Path,
    seconds: float = 10.0,
    fps: int = 25,
    *,
    start_seconds: float = 0.0,
) -> dict:
    if not 0 < seconds <= 60 or fps not in (10, 25, 50):
        raise ValueError("Render duration must be within (0, 60] seconds; fps must be 10, 25 or 50")
    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("Render start must be nonnegative and finite")
    if output.exists():
        raise FileExistsError(output)
    vendor = Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json")
    if not vendor.is_file():
        raise RuntimeError("CPU rendering requires the Mesa EGL vendor library")
    os.environ.update(
        MUJOCO_GL="egl",
        LIBGL_ALWAYS_SOFTWARE="1",
        __EGL_VENDOR_LIBRARY_FILENAMES=str(vendor),
        LP_NUM_THREADS="2",
    )
    # Import after selecting Mesa: mujoco/OpenGL choose their GL backend on import.
    # Select Mesa's software device before creating any display/context. The
    # default MuJoCo search tries hardware devices first, despite the software
    # environment flag, which can touch their VRAM or emit driver errors.
    import ctypes

    import imageio_ffmpeg
    import mujoco
    import numpy as np
    from mujoco.egl import egl_ext as EGL
    from OpenGL.GL import GL_RENDERER, glGetString
    from PIL import Image, ImageDraw

    address = EGL.eglGetProcAddress("eglQueryDeviceStringEXT")
    if not address:
        raise RuntimeError("EGL software-device query is unavailable")
    query_device = ctypes.CFUNCTYPE(ctypes.c_char_p, EGL.EGLDeviceEXT, EGL.EGLint)(address)
    software_device = next(
        (
            index
            for index, device in enumerate(EGL.eglQueryDevicesEXT())
            if b"EGL_MESA_device_software"
            in (query_device(device, EGL.EGL_EXTENSIONS) or b"").split()
        ),
        None,
    )
    if software_device is None:
        raise RuntimeError("No Mesa software EGL device is available")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(software_device)

    contract = json.loads((bundle / "contract.json").read_text())
    model_path = bundle / "model.mjb"
    if sha256(model_path) != contract["files"]["model.mjb"]:
        raise ValueError("Model hash differs from the export contract")
    with np.load(trajectory, allow_pickle=False) as data:
        states = data["states"].copy()
        dt = float(data["control_dt"])
    if states.ndim != 2 or not np.isfinite(states).all() or abs(dt - 0.02) > 1e-9:
        raise ValueError("Expected finite 50 Hz physical states")
    model = mujoco.MjModel.from_binary_path(str(model_path))
    if states.shape[1] != model.nq + model.nv:
        raise ValueError("Trajectory state dimension differs from model")
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.8, 130, -15
    root = contract["root_qpos_index"]
    frame_step = 50 // fps
    first_frame = round(start_seconds / dt)
    if first_frame >= len(states):
        raise ValueError("Render start is outside the saved physical trajectory")
    indices = range(first_frame, min(len(states), first_frame + round(seconds / dt)), frame_step)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    frames = 0
    with mujoco.Renderer(model, 480, 640) as renderer:
        backend = glGetString(GL_RENDERER).decode()
        if "llvmpipe" not in backend.lower() and "softpipe" not in backend.lower():
            raise RuntimeError(f"Expected software GL renderer, got {backend}")
        writer = imageio_ffmpeg.write_frames(
            str(output),
            (640, 480),
            fps=fps,
            codec="libx264",
            output_params=["-crf", "24", "-preset", "fast", "-threads", "2"],
        )
        writer.send(None)
        try:
            for index in indices:
                data.qpos[:] = states[index, : model.nq]
                data.qvel[:] = states[index, model.nq :]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = data.qpos[root : root + 3]
                renderer.update_scene(data, camera)
                picture = Image.fromarray(renderer.render())
                overlay = ImageDraw.Draw(picture)
                overlay.rectangle((0, 0, 640, 31), fill=(8, 18, 29))
                overlay.text(
                    (12, 10),
                    f"G1  |  iteration {contract['checkpoint_iteration']}  |  t = {index * dt:.2f} s  |  saved physical trajectory",
                    fill="white",
                )
                if frames == 0:
                    picture.save(output.with_suffix(".png"))
                writer.send(np.asarray(picture))
                frames += 1
        finally:
            writer.close()
    result = {
        "status": "complete",
        "renderer": backend,
        "egl_software_device_index": software_device,
        "cuda_used": False,
        "source": "saved actual physical states; no physics rerun or state modification",
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "trajectory_sha256": sha256(trajectory),
        "video_sha256": sha256(output),
        "frames": frames,
        "fps": fps,
        "first_source_frame": first_frame,
        "source_start_seconds": first_frame * dt,
        "duration_s": frames / fps,
        "elapsed_s": time.monotonic() - started,
    }
    write_json(output.with_suffix(".json"), result)
    return result
