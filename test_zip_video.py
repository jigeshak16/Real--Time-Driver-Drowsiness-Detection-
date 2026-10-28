
from pathlib import Path
from zipfile import ZipFile
import subprocess
import imageio_ffmpeg

zip_path = Path(
    r"C:\Users\Jigesha\Downloads\Fold3_part2.zip"
)
member = "Fold3_part2/32/10_1.mp4"

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
print("FFmpeg executable:", ffmpeg)

with ZipFile(zip_path, "r") as archive:
    video_bytes = archive.read(member)

print("Compressed video size:", len(video_bytes), "bytes")

command = [
    ffmpeg,
    "-hide_banner",
    "-loglevel", "error",
    "-i", "pipe:0",
    "-frames:v", "1",
    "-f", "image2pipe",
    "-vcodec", "mjpeg",
    "pipe:1",
]

try:
    result = subprocess.run(
        command,
        input=video_bytes,
        capture_output=True,
        timeout=60,
        check=False,
    )

    print("FFmpeg return code:", result.returncode)
    print("Decoded image bytes:", len(result.stdout))

    if result.stderr:
        print("FFmpeg message:")
        print(result.stderr.decode("utf-8", errors="replace")[:2000])

    if result.returncode == 0 and result.stdout:
        print("SUCCESS: FFmpeg decoded a frame directly from the ZIP video.")
    else:
        print("FAILED: FFmpeg could not decode this video.")

except subprocess.TimeoutExpired:
    print("FAILED: FFmpeg timed out.")

