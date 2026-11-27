"""Render saved physical trajectories with Mesa llvmpipe, without CUDA allocation."""

from __future__ import annotations

import fcntl
import math
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile

from supermimic.artifacts import sha256, snapshot_file, write_json

_GIF_MAX_SHADOW_SIZE = 2048


def _render_inputs(bundle: Path, trajectory: Path):
    """Read saved states without requiring controls, contact evidence or a result."""
    from supermimic.deployment.evidence import (
        read_json_evidence,
        read_optional_result,
        verify_artifact_identity,
        verify_control_dt,
    )
    from supermimic.deployment.trajectory import open_trajectory

    contract, contract_sha = read_json_evidence(bundle / "contract.json")
    with open_trajectory(trajectory) as recorded:
        if abs(recorded.control_dt - 0.02) > 1e-9:
            raise ValueError("Expected finite 50 Hz physical states")
    result, result_sha = read_optional_result(trajectory)
    identity = verify_artifact_identity(
        result,
        trajectory_sha256=recorded.sha256,
        contract_sha256=contract_sha,
        checkpoint_sha256=contract["checkpoint_sha256"],
    )
    verify_control_dt(recorded.control_dt, contract=contract, result=result)
    return (
        contract,
        recorded,
        {
            "contract_sha256": contract_sha,
            "artifact_identity": identity,
            "result_sha256": result_sha,
        },
    )


def _load_render_model(bundle: Path, contract: dict, output: Path, loader):
    """Give the native loader the verified MJB snapshot, then remove that copy."""
    existing = output.expanduser().resolve().parent
    while not existing.exists():
        existing = existing.parent
    with TemporaryDirectory(prefix=".supermimic-render-", dir=existing) as temporary:
        pinned_model = Path(temporary) / "model.mjb"
        snapshot_file(
            bundle / "model.mjb", pinned_model, expected_sha256=contract["files"]["model.mjb"]
        )
        return loader(str(pinned_model))


def _software_device() -> int:
    vendor = Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json")
    if not vendor.is_file():
        raise RuntimeError("CPU rendering requires the Mesa EGL vendor library")
    os.environ.update(
        MUJOCO_GL="egl",
        LIBGL_ALWAYS_SOFTWARE="1",
        __EGL_VENDOR_LIBRARY_FILENAMES=str(vendor),
        LP_NUM_THREADS="2",
    )
    # Select Mesa's software device before creating any display/context.
    import ctypes

    from mujoco.egl import egl_ext as EGL

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
    return software_device


def _frame_indices(state_count: int, dt: float, start: float, seconds: float, fps: int) -> range:
    if not math.isfinite(start) or not 0 <= start < state_count * dt:
        raise ValueError("Render start is outside the saved physical trajectory")
    first = round(start / dt)
    if first >= state_count:
        raise ValueError("Render start is outside the saved physical trajectory")
    count = max(1, round(seconds / dt))
    return range(first, min(state_count, first + count), 50 // fps)


def _ffmpeg_command() -> list[str]:
    from imageio_ffmpeg import get_ffmpeg_exe

    return [
        get_ffmpeg_exe(),
        "-nostdin",
        "-loglevel", "error",
        "-filter_threads", "1",
        "-filter_complex_threads", "1",
        "-threads", "1",
    ]


@contextmanager
def _frame_writer(path: Path, width: int, height: int, fps: int, *, lossless: bool):
    codec = (
        ["-c:v", "libx264rgb", "-preset", "ultrafast", "-crf", "0"]
        if lossless
        else ["-c:v", "libx264", "-preset", "fast", "-crf", "24", "-pix_fmt", "yuv420p"]
    )
    fast_start = (
        ["-movflags", "+faststart"] if not lossless and path.suffix in (".mp4", ".mov") else []
    )
    command = [
        *_ffmpeg_command(),
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", "pipe:0",
        *codec,
        "-threads", "1",
        *fast_start,
        str(path),
    ]
    with TemporaryFile() as errors, subprocess.Popen(
        command, stdin=subprocess.PIPE, stderr=errors
    ) as encoder:
        broken_pipe = False
        try:
            with encoder.stdin:
                yield encoder.stdin
        except BrokenPipeError:
            broken_pipe = True
        except BaseException:
            encoder.kill()
            raise
        try:
            code = encoder.wait(timeout=60)
        except subprocess.TimeoutExpired as error:
            encoder.kill()
            raise RuntimeError("FFmpeg timed out finalizing rendered frames") from error
        if code or broken_pipe:
            size = errors.seek(0, os.SEEK_END)
            errors.seek(max(0, size - 4096))
            detail = errors.read().decode(errors="replace").strip()
            raise RuntimeError(f"FFmpeg failed to encode rendered frames (exit {code}): {detail}")


def _encode_gif(source: Path, target: Path, fps: int) -> None:
    """Build one palette on disk so full RGB clips never accumulate in memory."""
    palette = source.with_name("palette.png")
    commands = (
        [
            "-i", str(source),
            "-vf", "palettegen=max_colors=64",
            "-frames:v", "1",
            "-threads", "1",
            str(palette),
        ],
        [
            "-i", str(source),
            "-threads", "1",
            "-i", str(palette),
            "-lavfi", "paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
            "-loop", "0",
            "-final_delay", str(100 // fps),
            "-threads", "1",
            str(target),
        ],
    )
    for arguments in commands:
        try:
            subprocess.run(
                [*_ffmpeg_command(), *arguments], check=True, capture_output=True, timeout=180
            )
        except subprocess.CalledProcessError as error:
            detail = error.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"FFmpeg failed to encode GIF: {detail}") from error
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("FFmpeg timed out encoding GIF") from error


@contextmanager
def _reserve_render_output(output: Path, poster: Path, metadata: Path):
    """Keep concurrent attempts from replacing even the resource report."""
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = metadata.with_name(f".{metadata.name}.render.lock")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise FileExistsError(f"Render already in progress for {output}") from None
        # A previous writer may have removed this inode between open and flock.
        if not os.path.samestat(lock_path.stat(), os.fstat(lock.fileno())):
            raise FileExistsError(f"Render destination changed for {output}")
        try:
            for path in (output, poster, metadata):
                if path.exists() or path.is_symlink():
                    raise FileExistsError(path)
            yield
        finally:
            lock_path.unlink()


def _publish_render(files: tuple[tuple[Path, Path], ...]) -> None:
    """Publish completed sidecars first, with the animation as the final file."""
    published = []
    try:
        for source, target in files:
            target.hardlink_to(source)
            published.append(target)
    except BaseException:
        for target in published:
            target.unlink()
        raise


def render_rollout(
    bundle: Path,
    trajectory: Path,
    output: Path,
    seconds: float = 10.0,
    fps: int | None = None,
    *,
    start_seconds: float = 0.0,
    width: int | None = None,
) -> dict:
    format_name = output.suffix.lower()
    if not format_name or format_name in (".png", ".json"):
        raise ValueError("Render output needs a video or GIF extension")
    gif = format_name == ".gif"
    if fps is None:
        fps = 10 if gif else 25
    if width is None:
        width = 480 if gif else 640
    if not 0 < seconds <= 60 or fps not in (10, 25, 50):
        raise ValueError("Render duration must be within (0, 60] seconds; fps must be 10, 25 or 50")
    if width not in (320, 384, 480, 640):
        raise ValueError("Render width must be 320, 384, 480 or 640")
    sidecar_base = output if gif else output.with_suffix("")
    poster = sidecar_base.with_name(f"{sidecar_base.name}.png")
    metadata = sidecar_base.with_name(f"{sidecar_base.name}.json")
    from supermimic.cpu_resources import cpu_guard
    from supermimic.resources import ResourceLimits

    with (
        _reserve_render_output(output, poster, metadata),
        TemporaryDirectory(prefix=".supermimic-render-", dir=output.parent) as directory,
    ):
        staged = Path(directory) / f"render{format_name}"
        staged_poster, staged_metadata = staged.with_suffix(".png"), staged.with_suffix(".json")
        with cpu_guard(output, limits=ResourceLimits(max_rss_gib=2)):
            result = _render_animation(
                bundle, trajectory, staged, staged_poster, seconds, fps, start_seconds, width
            )
            write_json(staged_metadata, result)
        _publish_render(((staged_poster, poster), (staged_metadata, metadata), (staged, output)))
    return result


def _render_animation(
    bundle: Path,
    trajectory: Path,
    staged: Path,
    poster: Path,
    seconds: float,
    fps: int,
    start_seconds: float,
    width: int,
) -> dict:
    gif = staged.suffix == ".gif"
    height = width * 3 // 4
    contract, recorded, evidence = _render_inputs(bundle, trajectory)
    states, dt = recorded.states, recorded.control_dt
    indices = _frame_indices(len(states), dt, start_seconds, seconds, fps)
    software_device = _software_device()

    # MuJoCo chooses its GL backend on import.
    import mujoco
    from OpenGL.GL import GL_RENDERER, glGetString
    from PIL import Image, ImageDraw

    model = _load_render_model(bundle, contract, staged, mujoco.MjModel.from_binary_path)
    if states.shape[1] != model.nq + model.nv:
        raise ValueError("Trajectory state dimension differs from model")
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.8, 130, -15
    root = contract["root_qpos_index"]
    if type(root) is not int or not 0 <= root <= model.nq - 3:
        raise ValueError("Root position index is outside the saved state layout")
    if gif:
        camera.distance, camera.elevation = 2.4, -27
        # Mesa allocates these buffers at the model's resolution even for a small preview.
        visual = model.vis
        visual.global_.offwidth = min(visual.global_.offwidth, width)
        visual.global_.offheight = min(visual.global_.offheight, height)
        visual.quality.shadowsize = min(visual.quality.shadowsize, _GIF_MAX_SHADOW_SIZE)
        # A plain floor keeps most pixels unchanged while the tracking camera moves.
        floor = model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
        model.geom_matid[floor] = -1
        model.geom_rgba[floor] = (0.16, 0.23, 0.30, 1)
    started = time.monotonic()

    intermediate = staged.with_name("frames.mp4") if gif else staged
    with (
        mujoco.Renderer(model, height, width) as renderer,
        _frame_writer(intermediate, width, height, fps, lossless=gif) as stream,
    ):
        backend = glGetString(GL_RENDERER).decode()
        if "llvmpipe" not in backend.lower() and "softpipe" not in backend.lower():
            raise RuntimeError(f"Expected software GL renderer, got {backend}")
        for index in indices:
            data.qpos[:] = states[index, : model.nq]
            data.qvel[:] = states[index, model.nq :]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.qpos[root : root + 3]
            if gif:
                # Keep small vertical movements visible and follow larger excursions.
                root_height = camera.lookat[2]
                camera.lookat[2] = min(max(0.8, root_height - 0.3), root_height + 0.3)
            renderer.update_scene(data, camera)
            if gif:
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            picture = Image.fromarray(renderer.render())
            overlay = ImageDraw.Draw(picture)
            overlay.rectangle((0, 0, width, 31), fill=(8, 18, 29))
            label = (
                f"G1  |  MuJoCo  |  t = {index * dt:.2f} s"
                if gif or width < 480
                else f"G1  |  iteration {contract['checkpoint_iteration']}  |  "
                f"t = {index * dt:.2f} s  |  saved trajectory"
            )
            overlay.text((12, 10), label, fill="white")
            if index == indices.start:
                picture.save(poster)
            stream.write(picture.tobytes())
    if gif:
        _encode_gif(intermediate, staged, fps)
    result = {
        "status": "complete",
        "renderer": backend,
        "egl_software_device_index": software_device,
        "cuda_used": False,
        "source": "saved trajectory states; no physics rerun or state modification",
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "trajectory_sha256": recorded.sha256,
        **evidence,
        "video_sha256": sha256(staged),
        "format": staged.suffix[1:],
        "width": width,
        "height": height,
        "frames": len(indices),
        "fps": fps,
        "first_source_frame": indices.start,
        "source_start_seconds": indices.start * dt,
        "source_end_seconds": indices[-1] * dt,
        "duration_s": len(indices) / fps,
        "elapsed_s": time.monotonic() - started,
    }
    return result
